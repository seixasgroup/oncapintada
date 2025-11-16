import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
import pytest
from ase import Atoms
from oncapintada.bondcounter import BondCounter  # Onça Pintada

@pytest.fixture # fixture to create a simple water molecule
def water_atoms() -> Atoms:
    return Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.74], [0.96, 0, 0]])


def test_bondcounter_init_requires_atoms_or_path():
    with pytest.raises(ValueError, match="Provide `atoms` or `xyz_path`."):
        BondCounter()


def test_bondcounter_init_with_atoms_sets_full_subset_and_target_pairs(water_atoms):
    counter = BondCounter(atoms=water_atoms)
    assert np.array_equal(counter._subset_mask, np.ones(len(water_atoms), dtype=bool))
    assert counter.target_pairs == {("H", "H"), ("H", "O"), ("O", "O")}


def test_bondcounter_init_subset_symbols_filters_mask_and_pairs(water_atoms):
    counter = BondCounter(atoms=water_atoms, subset_symbols=["H"])
    assert np.array_equal(counter._subset_mask, np.array([True, True, False]))
    assert counter.target_pairs == {("H", "H")}


def test_bondcounter_init_subset_indices_filters_mask(water_atoms):
    counter = BondCounter(atoms=water_atoms, subset_indices=[0, 2])
    assert np.array_equal(counter._subset_mask, np.array([True, False, True]))
    assert counter.target_pairs == {("H", "H"), ("H", "O"), ("O", "O")}


def test_set_subset_none_resets_to_all_atoms(water_atoms):
    counter = BondCounter(atoms=water_atoms, subset_indices=[0])
    counter.set_subset(None)
    assert np.array_equal(counter._subset_mask, np.ones(len(water_atoms), dtype=bool))


def test_set_subset_with_iterable(water_atoms):
    counter = BondCounter(atoms=water_atoms)
    counter.set_subset([0, 2])
    assert np.array_equal(counter._subset_mask, np.array([True, False, True]))


def test_set_subset_with_numpy_array_and_duplicates(water_atoms):
    counter = BondCounter(atoms=water_atoms)
    counter.set_subset(np.array([1, 1]))
    assert np.array_equal(counter._subset_mask, np.array([False, True, False]))



