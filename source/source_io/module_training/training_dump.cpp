#if defined(USE_LIBXC)

#include "training_dump.h"

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_global.h"
#include "source_base/timer.h"
#include "source_estate/module_charge/charge.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_hamilt/module_xc/libxc_abacus.h"
#include "source_io/module_parameter/parameter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef __MPI
#include <mpi.h>
#endif

namespace
{

constexpr double RY_TO_HA = 0.5;

double ry_to_ha(const double energy_ry)
{
    return RY_TO_HA * energy_ry;
}

bool is_norm_conserving_pseudo(const UnitCell& ucell)
{
    for (int it = 0; it < ucell.ntype; ++it)
    {
        if (ucell.atoms[it].ncpp.tvanp)
        {
            return false;
        }
        if (ucell.atoms[it].ncpp.pp_type != "NC")
        {
            return false;
        }
    }
    return true;
}

std::vector<double> gather_spin_major_to_global(const ModulePW::PW_Basis& rho_basis,
                                                const std::vector<double>& local,
                                                const int nchannels)
{
    const int nrxx_local = rho_basis.nrxx;
    const int nxyz = rho_basis.nxyz;
    if (nchannels <= 0)
    {
        return {};
    }

#ifdef __MPI
    if (GlobalV::NPROC_IN_POOL > 1 && POOL_WORLD != MPI_COMM_NULL)
    {
        const int ncxy = rho_basis.nx * rho_basis.ny;
        std::vector<double> global(nchannels * nxyz, 0.0);
        std::vector<double> gathered(nchannels * nxyz, 0.0);
        std::vector<int> rec(GlobalV::NPROC_IN_POOL);
        std::vector<int> dis(GlobalV::NPROC_IN_POOL);
        for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip)
        {
            rec[ip] = rho_basis.numz[ip] * ncxy;
            dis[ip] = rho_basis.startz[ip] * ncxy;
        }

        for (int ich = 0; ich < nchannels; ++ich)
        {
            MPI_Allgatherv(local.data() + ich * nrxx_local,
                           nrxx_local,
                           MPI_DOUBLE,
                           gathered.data() + ich * nxyz,
                           rec.data(),
                           dis.data(),
                           MPI_DOUBLE,
                           POOL_WORLD);

            double* global_ch = global.data() + ich * nxyz;
            const double* gathered_ch = gathered.data() + ich * nxyz;
            for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip)
            {
                for (int ixy = 0; ixy < ncxy; ++ixy)
                {
                    for (int iz = 0; iz < rho_basis.numz[ip]; ++iz)
                    {
                        global_ch[rho_basis.nz * ixy + rho_basis.startz[ip] + iz] =
                            gathered_ch[rho_basis.numz[ip] * ixy + rho_basis.startz[ip] * ncxy + iz];
                    }
                }
            }
        }
        return global;
    }
#endif

    return local;
}

std::string escape_json(const std::string& value)
{
    std::ostringstream os;
    for (const char c : value)
    {
        switch (c)
        {
        case '\\':
            os << "\\\\";
            break;
        case '"':
            os << "\\\"";
            break;
        case '\n':
            os << "\\n";
            break;
        case '\r':
            os << "\\r";
            break;
        case '\t':
            os << "\\t";
            break;
        default:
            os << c;
        }
    }
    return os.str();
}

std::string json_string(const std::string& value)
{
    return "\"" + escape_json(value) + "\"";
}

void write_npy_header(std::ofstream& os, const std::vector<unsigned long>& shape)
{
    std::ostringstream dict;
    dict << "{'descr': '<f8', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            dict << ", ";
        }
        dict << shape[i];
    }
    if (shape.size() == 1)
    {
        dict << ",";
    }
    dict << "), }";

    std::string header = dict.str();
    const std::size_t prefix_size = 10;
    const std::size_t padding = 16 - ((prefix_size + header.size() + 1) % 16);
    header.append(padding, ' ');
    header.push_back('\n');

    os.write("\x93NUMPY", 6);
    const char version[2] = {1, 0};
    os.write(version, 2);
    const unsigned short header_len = static_cast<unsigned short>(header.size());
    os.put(static_cast<char>(header_len & 0xff));
    os.put(static_cast<char>((header_len >> 8) & 0xff));
    os.write(header.data(), static_cast<std::streamsize>(header.size()));
}

void write_npy(const std::string& filename,
               const std::vector<unsigned long>& shape,
               const std::vector<double>& data)
{
    std::ofstream os(filename.c_str(), std::ios::binary);
    if (!os)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "Failed to open " + filename);
    }
    write_npy_header(os, shape);
    if (!data.empty())
    {
        os.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size() * sizeof(double)));
    }
}

std::vector<double> matrix_to_vector(const ModuleBase::matrix& matrix)
{
    const int size = matrix.nr * matrix.nc;
    std::vector<double> out(size, 0.0);
    for (int i = 0; i < size; ++i)
    {
        out[i] = matrix.c[i];
    }
    return out;
}

std::vector<double> compute_sigma_local(const int nspin,
                                        const int nsigma,
                                        const int nrxx,
                                        const std::vector<double>& rho_interleaved,
                                        const double tpiba,
                                        const Charge& chr)
{
    const auto gdr = XC_Functional_Libxc::cal_gdr(nspin, nrxx, rho_interleaved, tpiba, &chr);
    const auto sigma_interleaved = XC_Functional_Libxc::convert_sigma(gdr);
    std::vector<double> sigma_local(nsigma * nrxx, 0.0);
    for (int isig = 0; isig < nsigma; ++isig)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            sigma_local[isig * nrxx + ir] = sigma_interleaved[ir * nsigma + isig];
        }
    }
    return sigma_local;
}

std::vector<double> ry_matrix_to_ha_vector(const ModuleBase::matrix& matrix)
{
    std::vector<double> out = matrix_to_vector(matrix);
    for (double& value : out)
    {
        value = ry_to_ha(value);
    }
    return out;
}

void compute_frontier(const elecstate::ElecState& elec, double& vbm, double& cbm)
{
    vbm = -std::numeric_limits<double>::infinity();
    cbm = std::numeric_limits<double>::infinity();
    if (elec.ekb.nr == 0 || elec.ekb.nc == 0)
    {
        vbm = elec.eferm.ef;
        cbm = elec.eferm.ef;
        return;
    }

    const int nks = elec.klist == nullptr ? elec.ekb.nr : elec.klist->get_nks();
    constexpr double threshold = 1.0e-5;
    for (int ik = 0; ik < nks; ++ik)
    {
        for (int ib = 0; ib < elec.ekb.nc; ++ib)
        {
            const double eig = elec.ekb(ik, ib);
            if (eig <= elec.eferm.ef + threshold && eig > vbm)
            {
                vbm = eig;
            }
            if (eig > elec.eferm.ef + threshold && eig < cbm)
            {
                cbm = eig;
            }
        }
    }
    if (cbm == std::numeric_limits<double>::infinity())
    {
        cbm = elec.eferm.ef;
    }
    if (vbm == -std::numeric_limits<double>::infinity())
    {
        vbm = elec.eferm.ef;
    }
#ifdef __MPI
    if (POOL_WORLD != MPI_COMM_NULL)
    {
        MPI_Allreduce(MPI_IN_PLACE, &vbm, 1, MPI_DOUBLE, MPI_MAX, POOL_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &cbm, 1, MPI_DOUBLE, MPI_MIN, POOL_WORLD);
    }
#endif
}

std::vector<double> kvector_matrix(const elecstate::ElecState& elec, const bool cartesian)
{
    std::vector<double> out;
    if (elec.klist == nullptr)
    {
        return out;
    }
    const int nks = elec.klist->get_nks();
    out.resize(nks * 3, 0.0);
    for (int ik = 0; ik < nks; ++ik)
    {
        const auto& kvec = cartesian ? elec.klist->kvec_c[ik] : elec.klist->kvec_d[ik];
        out[3 * ik] = kvec.x;
        out[3 * ik + 1] = kvec.y;
        out[3 * ik + 2] = kvec.z;
    }
    return out;
}

std::vector<double> kweights(const elecstate::ElecState& elec)
{
    std::vector<double> out;
    if (elec.klist == nullptr)
    {
        return out;
    }
    const int nks = elec.klist->get_nks();
    out.resize(nks, 0.0);
    for (int ik = 0; ik < nks; ++ik)
    {
        out[ik] = elec.klist->wk[ik];
    }
    return out;
}

std::vector<double> kspins(const elecstate::ElecState& elec)
{
    std::vector<double> out;
    if (elec.klist == nullptr)
    {
        return out;
    }
    const int nks = elec.klist->get_nks();
    out.resize(nks, 0.0);
    for (int ik = 0; ik < nks; ++ik)
    {
        out[ik] = static_cast<double>(elec.klist->isk[ik]);
    }
    return out;
}

} // namespace

namespace ModuleIO
{

void write_training_dump(const UnitCell& ucell,
                         const elecstate::ElecState& elec,
                         const ModulePW::PW_Basis& rho_basis,
                         const Charge& chr,
                         const int istep,
                         const std::string& output_root)
{
    ModuleBase::TITLE("ModuleIO", "write_training_dump");
    ModuleBase::timer::start("ModuleIO", "write_training_dump");

    if (PARAM.inp.basis_type != "pw")
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data is only available for basis_type pw");
    }
    std::string dft_functional_lower = PARAM.inp.dft_functional;
    std::transform(dft_functional_lower.begin(),
                   dft_functional_lower.end(),
                   dft_functional_lower.begin(),
                   ::tolower);
    if (dft_functional_lower != "pbe")
    {
        ModuleBase::WARNING_QUIT("write_training_dump",
                                 "out_training_data currently requires dft_functional = PBE for the PBE-base EXX label workflow");
    }
    if (PARAM.inp.nspin != 1 && PARAM.inp.nspin != 2)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data supports nspin 1 or 2");
    }
    if (PARAM.inp.kpar != 1)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data currently requires kpar = 1");
    }
    if (PARAM.inp.bndpar != 1)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data currently requires bndpar = 1");
    }
    if (!is_norm_conserving_pseudo(ucell))
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "out_training_data currently requires NC pseudopotentials");
    }

    const int nspin = PARAM.inp.nspin;
    const int nsigma = (nspin == 1) ? 1 : 3;
    const int nrxx = rho_basis.nrxx;
    const int nxyz = rho_basis.nxyz;
    const double dvol = ucell.omega / static_cast<double>(nxyz);

    std::vector<double> rho_valence_local(nspin * nrxx, 0.0);
    std::vector<double> rho_total_local(nspin * nrxx, 0.0);
    std::vector<double> rho_valence_interleaved(nspin * nrxx, 0.0);
    std::vector<double> rho_interleaved(nspin * nrxx, 0.0);
    for (int is = 0; is < nspin; ++is)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            const double rho_valence = chr.rho[is][ir];
            const double rho_total = rho_valence + chr.rho_core[ir] / static_cast<double>(nspin);
            rho_valence_local[is * nrxx + ir] = rho_valence;
            rho_total_local[is * nrxx + ir] = rho_total;
            rho_valence_interleaved[ir * nspin + is] = rho_valence;
            rho_interleaved[ir * nspin + is] = rho_total;
        }
    }

    std::vector<double> rho_core_local(nrxx, 0.0);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        rho_core_local[ir] = chr.rho_core[ir];
    }

    const std::vector<double> sigma_valence_local =
        compute_sigma_local(nspin, nsigma, nrxx, rho_valence_interleaved, ucell.tpiba, chr);
    const std::vector<double> sigma_local =
        compute_sigma_local(nspin, nsigma, nrxx, rho_interleaved, ucell.tpiba, chr);

    std::vector<double> tau_local;
    std::vector<double> tau_valence_local;
    if (chr.kin_r != nullptr)
    {
        tau_local.assign(nspin * nrxx, 0.0);
        tau_valence_local.assign(nspin * nrxx, 0.0);
        const double pi = std::acos(-1.0);
        const double tf_factor = (3.0 / 10.0) * std::pow(3.0 * pi * pi, 2.0 / 3.0);
        for (int is = 0; is < nspin; ++is)
        {
            for (int ir = 0; ir < nrxx; ++ir)
            {
                double tau = chr.kin_r[is][ir] / 2.0;
                tau_valence_local[is * nrxx + ir] = tau;
                if (PARAM.inp.cider_tf_tau)
                {
                    const double rho_cps = std::max(chr.rho_core[ir] / static_cast<double>(nspin), 0.0);
                    tau += tf_factor * std::pow(rho_cps, 5.0 / 3.0);
                }
                tau_local[is * nrxx + ir] = tau;
            }
        }
    }

    const std::vector<double> rho_valence = gather_spin_major_to_global(rho_basis, rho_valence_local, nspin);
    const std::vector<double> rho_total = gather_spin_major_to_global(rho_basis, rho_total_local, nspin);
    const std::vector<double> rho_core = gather_spin_major_to_global(rho_basis, rho_core_local, 1);
    const std::vector<double> sigma = gather_spin_major_to_global(rho_basis, sigma_local, nsigma);
    const std::vector<double> sigma_valence = gather_spin_major_to_global(rho_basis, sigma_valence_local, nsigma);
    const std::vector<double> tau = tau_local.empty() ? std::vector<double>()
                                                      : gather_spin_major_to_global(rho_basis, tau_local, nspin);
    const std::vector<double> tau_valence = tau_valence_local.empty()
                                                ? std::vector<double>()
                                                : gather_spin_major_to_global(rho_basis, tau_valence_local, nspin);

    double vbm = 0.0;
    double cbm = 0.0;
    compute_frontier(elec, vbm, cbm);

#ifdef __MPI
    if (GlobalV::MY_RANK != 0)
    {
        ModuleBase::timer::end("ModuleIO", "write_training_dump");
        return;
    }
#endif

    const std::string base_dir = output_root + "training_dump";
    ModuleBase::GlobalFunc::MAKE_DIR(base_dir);
    const std::string dump_dir = base_dir + "/step_" + std::to_string(std::max(istep, 0));
    ModuleBase::GlobalFunc::MAKE_DIR(dump_dir);

    write_npy(dump_dir + "/rho_valence_sg.npy",
              {static_cast<unsigned long>(nspin), static_cast<unsigned long>(nxyz)},
              rho_valence);
    write_npy(dump_dir + "/rho_sg.npy",
              {static_cast<unsigned long>(nspin), static_cast<unsigned long>(nxyz)},
              rho_total);
    write_npy(dump_dir + "/rho_core_g.npy", {static_cast<unsigned long>(nxyz)}, rho_core);
    write_npy(dump_dir + "/sigma_xg.npy",
              {static_cast<unsigned long>(nsigma), static_cast<unsigned long>(nxyz)},
              sigma);
    write_npy(dump_dir + "/sigma_valence_xg.npy",
              {static_cast<unsigned long>(nsigma), static_cast<unsigned long>(nxyz)},
              sigma_valence);
    if (!tau.empty())
    {
        write_npy(dump_dir + "/tau_sg.npy",
                  {static_cast<unsigned long>(nspin), static_cast<unsigned long>(nxyz)},
                  tau);
        write_npy(dump_dir + "/tau_valence_sg.npy",
                  {static_cast<unsigned long>(nspin), static_cast<unsigned long>(nxyz)},
                  tau_valence);
    }

    write_npy(dump_dir + "/ekb_kb.npy",
              {static_cast<unsigned long>(elec.ekb.nr), static_cast<unsigned long>(elec.ekb.nc)},
              ry_matrix_to_ha_vector(elec.ekb));
    write_npy(dump_dir + "/occ_kb.npy",
              {static_cast<unsigned long>(elec.wg.nr), static_cast<unsigned long>(elec.wg.nc)},
              matrix_to_vector(elec.wg));

    if (elec.klist != nullptr)
    {
        const int nks = elec.klist->get_nks();
        write_npy(dump_dir + "/kpt_cart_kv.npy",
                  {static_cast<unsigned long>(nks), 3},
                  kvector_matrix(elec, true));
        write_npy(dump_dir + "/kpt_direct_kv.npy",
                  {static_cast<unsigned long>(nks), 3},
                  kvector_matrix(elec, false));
        write_npy(dump_dir + "/wk_k.npy", {static_cast<unsigned long>(nks)}, kweights(elec));
        write_npy(dump_dir + "/kspin_k.npy", {static_cast<unsigned long>(nks)}, kspins(elec));
    }

    std::ofstream js((dump_dir + "/record.json").c_str());
    if (!js)
    {
        ModuleBase::WARNING_QUIT("write_training_dump", "Failed to open record.json");
    }
    js << std::setprecision(16);
    js << "{\n";
    js << "  \"schema_name\": \"abacus_pw_nldx_training_dump\",\n";
    js << "  \"schema_version\": 2,\n";
    js << "  \"record_kind\": \"pbe_base_state\",\n";
    js << "  \"label_mode\": \"pbe_base_state_one_shot_pbe0_exx_label\",\n";
    js << "  \"energy_unit\": \"Ha\",\n";
    js << "  \"length_unit\": \"Bohr\",\n";
    js << "  \"density_unit\": \"Bohr^-3\",\n";
    js << "  \"sigma_unit\": \"Bohr^-8\",\n";
    js << "  \"tau_unit\": \"Bohr^-5\",\n";
    js << "  \"kpt_cart_unit\": \"Bohr^-1\",\n";
    js << "  \"basis_type\": " << json_string(PARAM.inp.basis_type) << ",\n";
    js << "  \"pseudo_type_required\": \"ncp\",\n";
    js << "  \"parallel_policy\": \"kpar_1_bndpar_1_required\",\n";
    js << "  \"source_state\": \"self_consistent_pbe_base\",\n";
    js << "  \"exx_energy_evaluated\": true,\n";
    js << "  \"exx_energy_source\": \"one_shot_pbe0_full_range_exx_on_pbe_state\",\n";
    js << "  \"nspin\": " << nspin << ",\n";
    js << "  \"nsigma\": " << nsigma << ",\n";
    js << "  \"nbands\": " << elec.ekb.nc << ",\n";
    js << "  \"nelec\": " << PARAM.inp.nelec << ",\n";
    js << "  \"grid_shape\": [" << rho_basis.nx << ", " << rho_basis.ny << ", " << rho_basis.nz << "],\n";
    js << "  \"nxyz\": " << nxyz << ",\n";
    js << "  \"dV_bohr3\": " << dvol << ",\n";
    js << "  \"volume_bohr3\": " << ucell.omega << ",\n";
    js << "  \"cell_cv_bohr\": [";
    js << ucell.lat0 * ucell.latvec.e11 << ", " << ucell.lat0 * ucell.latvec.e12 << ", "
       << ucell.lat0 * ucell.latvec.e13 << ", " << ucell.lat0 * ucell.latvec.e21 << ", "
       << ucell.lat0 * ucell.latvec.e22 << ", " << ucell.lat0 * ucell.latvec.e23 << ", "
       << ucell.lat0 * ucell.latvec.e31 << ", " << ucell.lat0 * ucell.latvec.e32 << ", "
       << ucell.lat0 * ucell.latvec.e33 << "],\n";
    js << "  \"ecutwfc_Ha\": " << ry_to_ha(PARAM.inp.ecutwfc) << ",\n";
    js << "  \"ecutrho_Ha\": " << ry_to_ha(PARAM.inp.ecutrho) << ",\n";
    js << "  \"dft_functional\": " << json_string(PARAM.inp.dft_functional) << ",\n";
    js << "  \"scf_thr\": " << PARAM.inp.scf_thr << ",\n";
    js << "  \"scf_nmax\": " << PARAM.inp.scf_nmax << ",\n";
    js << "  \"scf_iter\": " << elec.iter << ",\n";
    js << "  \"converged\": " << (elec.iter <= PARAM.inp.scf_nmax ? "true" : "false") << ",\n";
    js << "  \"ionic_step\": " << istep << ",\n";
    js << "  \"cider_model\": " << json_string(PARAM.inp.cider_model) << ",\n";
    js << "  \"cider_xmix\": " << PARAM.inp.cider_xmix << ",\n";
    js << "  \"cider_tf_tau\": " << (PARAM.inp.cider_tf_tau ? "true" : "false") << ",\n";
    js << "  \"tau_present\": " << (!tau.empty() ? "true" : "false") << ",\n";
    js << "  \"efermi_Ha\": " << ry_to_ha(elec.eferm.ef) << ",\n";
    js << "  \"vbm_Ha\": " << ry_to_ha(vbm) << ",\n";
    js << "  \"cbm_Ha\": " << ry_to_ha(cbm) << ",\n";
    js << "  \"bandgap_Ha\": " << ry_to_ha(cbm - vbm) << ",\n";
    js << "  \"energy_Ha\": {\n";
    js << "    \"etot\": " << ry_to_ha(elec.f_en.etot) << ",\n";
    js << "    \"eband\": " << ry_to_ha(elec.f_en.eband) << ",\n";
    js << "    \"deband\": " << ry_to_ha(elec.f_en.deband) << ",\n";
    js << "    \"etxc\": " << ry_to_ha(elec.f_en.etxc) << ",\n";
    js << "    \"vtxc\": " << ry_to_ha(elec.f_en.vtxc) << ",\n";
    js << "    \"hartree\": " << ry_to_ha(elec.f_en.hartree_energy) << ",\n";
    js << "    \"ewald\": " << ry_to_ha(elec.f_en.ewald_energy) << ",\n";
    js << "    \"local_pp\": " << ry_to_ha(elec.f_en.e_local_pp) << "\n";
    js << "  },\n";
    js << "  \"exx_label\": {\n";
    js << "    \"present\": true,\n";
    js << "    \"definition\": \"one-shot PBE0 full-range EXX energy evaluated on the converged PBE density and plane-wave wavefunctions\",\n";
    js << "    \"energy_evaluated\": true,\n";
    js << "    \"energy_source\": \"one_shot_pbe0_full_range_exx_on_pbe_state\",\n";
    js << "    \"energy_Ha\": " << ry_to_ha(elec.f_en.exx) << ",\n";
    js << "    \"raw_exx_energy_available\": false,\n";
    js << "    \"hybrid_scaled\": true,\n";
    js << "    \"hybrid_alpha\": 0.25,\n";
    js << "    \"range\": \"full\",\n";
    js << "    \"screening\": \"none\",\n";
    js << "    \"base_state\": \"pbe_scf_density_and_pw_wavefunctions\",\n";
    js << "    \"self_consistent_hybrid_required\": false\n";
    js << "  },\n";
    js << "  \"arrays\": {\n";
    js << "    \"rho_valence_sg\": \"rho_valence_sg.npy\",\n";
    js << "    \"rho_sg\": \"rho_sg.npy\",\n";
    js << "    \"rho_core_g\": \"rho_core_g.npy\",\n";
    js << "    \"sigma_xg\": \"sigma_xg.npy\",\n";
    js << "    \"sigma_valence_xg\": \"sigma_valence_xg.npy\",\n";
    if (!tau.empty())
    {
        js << "    \"tau_sg\": \"tau_sg.npy\",\n";
        js << "    \"tau_valence_sg\": \"tau_valence_sg.npy\",\n";
    }
    js << "    \"ekb_kb\": \"ekb_kb.npy\",\n";
    js << "    \"occ_kb\": \"occ_kb.npy\",\n";
    js << "    \"kpt_cart_kv\": \"kpt_cart_kv.npy\",\n";
    js << "    \"kpt_direct_kv\": \"kpt_direct_kv.npy\",\n";
    js << "    \"wk_k\": \"wk_k.npy\",\n";
    js << "    \"kspin_k\": \"kspin_k.npy\"\n";
    js << "  },\n";
    js << "  \"pseudopotentials\": [\n";
    for (int it = 0; it < ucell.ntype; ++it)
    {
        js << "    {\"label\": " << json_string(ucell.atoms[it].label)
           << ", \"file\": " << json_string(it < static_cast<int>(ucell.pseudo_fn.size()) ? ucell.pseudo_fn[it] : "")
           << ", \"type\": " << json_string(it < static_cast<int>(ucell.pseudo_type.size()) ? ucell.pseudo_type[it] : "")
           << ", \"nlcc\": " << (ucell.atoms[it].ncpp.nlcc ? "true" : "false") << "}";
        js << (it + 1 == ucell.ntype ? "\n" : ",\n");
    }
    js << "  ]\n";
    js << "}\n";

    GlobalV::ofs_running << "write_training_dump: wrote " << dump_dir << std::endl;
    ModuleBase::timer::end("ModuleIO", "write_training_dump");
}

} // namespace ModuleIO

#endif
