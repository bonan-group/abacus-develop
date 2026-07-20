#include "source_cell/atom_spec.h"
#include "source_cell/setup_nonlocal.h"
#include "source_cell/magnetism.h"

pseudo::pseudo() = default;
pseudo::~pseudo() = default;

Atom_pseudo::Atom_pseudo() = default;
Atom_pseudo::~Atom_pseudo() = default;

Atom::Atom() = default;
Atom::~Atom() = default;

InfoNonlocal::InfoNonlocal() = default;
InfoNonlocal::~InfoNonlocal() = default;

Magnetism::Magnetism()
{
    tot_mag = 0.0;
    abs_mag = 0.0;
}

Magnetism::~Magnetism() = default;
