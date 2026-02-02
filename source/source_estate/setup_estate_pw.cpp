#include "source_estate/setup_estate_pw.h"
#include "source_estate/elecstate_pw.h" // init of pelec
#include "source_estate/elecstate_pw_sdft.h" // init of pelec for sdft
#include "source_estate/elecstate_tools.h" // occupations
#include "source_pw/module_pwdft/op_pw_nl.h" // use_chunked_vnl

template <typename T, typename Device>
void elecstate::setup_estate_pw(UnitCell& ucell, // unitcell
		K_Vectors &kv, // kpoints
        Structure_Factor &sf, // structure factors
		elecstate::ElecState* &pelec, // pointer of electrons
		Charge &chr, // charge density
		pseudopot_cell_vl &locpp, // local pseudopotentials
		pseudopot_cell_vnl &ppcell, // non-local pseudopotentials
		VSep* &vsep_cell, // U-1/2 method
		ModulePW::PW_Basis_K* pw_wfc,  // pw for wfc
		ModulePW::PW_Basis* pw_rho, // pw for rho
		ModulePW::PW_Basis* pw_rhod, // pw for rhod
        ModulePW::PW_Basis_Big* pw_big, // pw for big grid
        surchem &solvent, //  solvent
		const Input_para& inp) // input parameters
{
    ModuleBase::TITLE("elecstate", "setup_estate_pw");

    //! Initialize ElecState, set pelec pointer
    if (pelec == nullptr)
    {
        if (inp.esolver_type == "sdft")
        {
            //! SDFT only supports double precision currently
            pelec = new elecstate::ElecStatePW_SDFT<std::complex<double>, Device>(pw_wfc,
                &chr, &kv, &ucell, &ppcell, pw_rho, pw_big);
        }
        else
        {
            pelec = new elecstate::ElecStatePW<T, Device>(pw_wfc,
                &chr, &kv, &ucell, &ppcell, pw_rho, pw_big);
        }
    }

    //! Initialize DFT-1/2
    if (PARAM.inp.dfthalf_type > 0)
    {
        vsep_cell = new VSep;
        vsep_cell->init_vsep(*pw_rhod, ucell.sep_cell);
    }

    //! Initialize the potential.
    if (pelec->pot == nullptr)
    {
        pelec->pot = new elecstate::Potential(pw_rhod,
              pw_rho, &ucell, &locpp.vloc, &sf,
              &solvent, &(pelec->f_en.etxc), &(pelec->f_en.vtxc), vsep_cell);
    }

    //! Initalize local pseudopotential
    locpp.init_vloc(ucell, pw_rhod);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "LOCAL POTENTIAL");

    //! Initalize non-local pseudopotential
    // When chunked VNL processing is enabled, skip full vkb allocation to save some memory
    // This reduces the overhead of storing a large array that scale as natoms^2 (nprojects * npw)
    // not a huge problem for CPU as the plane waves are distributed (with g-vector decomposition)
    // but can be a problem for GPU with limited memory when the system size is large
    // use_chunked_vnl checks: 1) env override, 2) CPU=off, 3) GPU=on if vkb>4GB
    const bool use_chunked = hamilt::use_chunked_vnl<T, Device>(ucell, pw_wfc);
    const bool allocate_vkb = !use_chunked;

    // Log chunked VNL status
    if (use_chunked)
    {
        const int nkb = hamilt::calculate_nkb(ucell);
        const int npwx = pw_wfc->npwk_max;
        const size_t vkb_size_mb = static_cast<size_t>(nkb) * npwx * sizeof(T) / (1024 * 1024);
        const int chunk_size = hamilt::get_chunk_size_override() > 0
                             ? hamilt::get_chunk_size_override()
                             : 64;  // default chunk size

        std::cout << " NOTICE: Chunked VNL processing ENABLED" << std::endl;
        std::cout << "         expected vkb size: " << vkb_size_mb << " MB, chunk size: " << chunk_size << " projectors" << std::endl;

        GlobalV::ofs_running << "\n NOTICE: Chunked VNL processing ENABLED" << std::endl;
        GlobalV::ofs_running << "         expected vkb size: " << vkb_size_mb << " MB, chunk size: " << chunk_size << " projectors" << std::endl;
    }

    ppcell.init(ucell, &sf, pw_wfc, allocate_vkb);
    ppcell.init_vnl(ucell, pw_rhod);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "NON-LOCAL POTENTIAL");

    //! Setup occupations
    if (inp.ocp)
    {
        elecstate::fixed_weights(inp.ocp_kb,
                                 inp.nbands,
                                 inp.nelec,
                                 pelec->klist,
                                 pelec->wg,
                                 pelec->skip_weights);
    }

    return;
}


template <typename T, typename Device>
void elecstate::teardown_estate_pw(elecstate::ElecState* &pelec, VSep* &vsep_cell) 
{
    ModuleBase::TITLE("elecstate", "teardown_estate_pw");

    if (vsep_cell != nullptr)
    {
        delete vsep_cell;
    }

    // mohan update 20251005 to increase the security level
    if (pelec != nullptr)
    {
		auto* pw_elec = dynamic_cast<elecstate::ElecStatePW<T, Device>*>(pelec);
		if (pw_elec) 
		{
			delete pw_elec;
			pelec = nullptr;
		} 
		else 
		{
            ModuleBase::WARNING_QUIT("elecstate::teardown_estate_pw", "Invalid ElecState type");
        }
    }
}


template void elecstate::setup_estate_pw<std::complex<float>, base_device::DEVICE_CPU>(
        UnitCell& ucell, // unitcell
		K_Vectors &kv, // kpoints
        Structure_Factor &sf, // structure factors
		elecstate::ElecState* &pelec, // pointer of electrons
		Charge &chr, // charge density
		pseudopot_cell_vl &locpp, // local pseudopotentials
		pseudopot_cell_vnl &ppcell, // non-local pseudopotentials
		VSep* &vsep_cell, // U-1/2 method
		ModulePW::PW_Basis_K *pw_wfc,  // pw for wfc
		ModulePW::PW_Basis *pw_rho, // pw for rho
		ModulePW::PW_Basis *pw_rhod, // pw for rhod
        ModulePW::PW_Basis_Big* pw_big, // pw for big grid
        surchem &solvent, //  solvent
		const Input_para& inp); // input parameters

template void elecstate::setup_estate_pw<std::complex<double>, base_device::DEVICE_CPU>(
        UnitCell& ucell, // unitcell
		K_Vectors &kv, // kpoints
        Structure_Factor &sf, // structure factors
		elecstate::ElecState* &pelec, // pointer of electrons
		Charge &chr, // charge density
		pseudopot_cell_vl &locpp, // local pseudopotentials
		pseudopot_cell_vnl &ppcell, // non-local pseudopotentials
		VSep* &vsep_cell, // U-1/2 method
		ModulePW::PW_Basis_K *pw_wfc,  // pw for wfc
		ModulePW::PW_Basis *pw_rho, // pw for rho
		ModulePW::PW_Basis *pw_rhod, // pw for rhod
        ModulePW::PW_Basis_Big* pw_big, // pw for big grid
        surchem &solvent, //  solvent
		const Input_para& inp); // input parameters


template void elecstate::teardown_estate_pw<std::complex<float>, base_device::DEVICE_CPU>(
        elecstate::ElecState* &pelec, VSep* &vsep_cell); 

template void elecstate::teardown_estate_pw<std::complex<double>, base_device::DEVICE_CPU>(
        elecstate::ElecState* &pelec, VSep* &vsep_cell); 


#if ((defined __CUDA) || (defined __ROCM))

template void elecstate::setup_estate_pw<std::complex<float>, base_device::DEVICE_GPU>(
        UnitCell& ucell, // unitcell
		K_Vectors &kv, // kpoints
        Structure_Factor &sf, // structure factors
		elecstate::ElecState* &pelec, // pointer of electrons
		Charge &chr, // charge density
		pseudopot_cell_vl &locpp, // local pseudopotentials
		pseudopot_cell_vnl &ppcell, // non-local pseudopotentials
		VSep* &vsep_cell, // U-1/2 method
		ModulePW::PW_Basis_K *pw_wfc,  // pw for wfc
		ModulePW::PW_Basis *pw_rho, // pw for rho
		ModulePW::PW_Basis *pw_rhod, // pw for rhod
        ModulePW::PW_Basis_Big* pw_big, // pw for big grid
        surchem &solvent, //  solvent
		const Input_para& inp); // input parameters

template void elecstate::setup_estate_pw<std::complex<double>, base_device::DEVICE_GPU>(
        UnitCell& ucell, // unitcell
		K_Vectors &kv, // kpoints
        Structure_Factor &sf, // structure factors
		elecstate::ElecState* &pelec, // pointer of electrons
		Charge &chr, // charge density
		pseudopot_cell_vl &locpp, // local pseudopotentials
		pseudopot_cell_vnl &ppcell, // non-local pseudopotentials
		VSep* &vsep_cell, // U-1/2 method
		ModulePW::PW_Basis_K *pw_wfc,  // pw for wfc
		ModulePW::PW_Basis *pw_rho, // pw for rho
		ModulePW::PW_Basis *pw_rhod, // pw for rhod
        ModulePW::PW_Basis_Big* pw_big, // pw for big grid
        surchem &solvent, //  solvent
		const Input_para& inp); // input parameters

template void elecstate::teardown_estate_pw<std::complex<float>, base_device::DEVICE_GPU>(
        elecstate::ElecState* &pelec, VSep* &vsep_cell); 

template void elecstate::teardown_estate_pw<std::complex<double>, base_device::DEVICE_GPU>(
        elecstate::ElecState* &pelec, VSep* &vsep_cell); 

#endif
