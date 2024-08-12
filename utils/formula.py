import re
from rdkit import Chem

class Formula:

    def __init__(self, formula: str):
        self.dict_representation = self.get_atom_and_counts(formula)

    def get_atom_and_counts(self, formula):
        parts = re.findall("[A-Z][a-z]?|[0-9]+", formula)
        atoms_and_counts = {}
        for i, atom in enumerate(parts):
            if atom.isnumeric():
                continue
            multiplier = int(parts[i + 1]) if len(parts) > i + 1 and parts[i + 1].isnumeric() else 1
            if atom in atoms_and_counts.keys():
                atoms_and_counts[atom] += multiplier
                # print(f"Repetition in formula found, {atom} occurs multiple times in {formula}")
            else:
                atoms_and_counts[atom] = multiplier
        return atoms_and_counts
        
    def __add__(self, otherFormula: "Formula"):
        new_formula = Formula("")
        new_formula.dict_representation = self.dict_representation.copy()
        for atom, value in otherFormula.dict_representation.items():
            if atom in new_formula.dict_representation:
                new_formula.dict_representation[atom] += value
            else:
                new_formula.dict_representation[atom] = value
        return new_formula
        
    def __sub__(self, otherFormula: "Formula"):
        new_formula = Formula("")
        new_formula.dict_representation = self.dict_representation.copy()
        for atom, value in otherFormula.dict_representation.items():
            if atom in new_formula.dict_representation:
                new_formula.dict_representation[atom] -= value
                if new_formula.dict_representation[atom] < 0:
                    print(f"Removing an atom {otherFormula} that does not exist in the main formula {str(self)}")
                    return None
            else:
                print(f"Removing an atom {otherFormula} that does not exist in the main formula {str(self)}")
                return None
        return new_formula

    def __mul__(self, multiplication):
        new_formula = Formula("")
        for i in range(multiplication):
            new_formula += self
        return new_formula
        
    def __str__(self):
        # Separate out carbon, hydrogen, and other elements
        carbon_count = self.dict_representation.get('C', 0)
        hydrogen_count = self.dict_representation.get('H', 0)
        
        # Elements except C and H
        other_elements = {k: v for k, v in self.dict_representation.items() if k not in ['C', 'H']}
        
        # Sort other elements alphabetically
        sorted_other_elements = sorted(other_elements.items())
        
        # Build the Hill notation string
        hill_notation = ''
        
        # Add carbon if it exists
        if carbon_count > 0:
            hill_notation += 'C'
            if carbon_count > 1:
                hill_notation += str(carbon_count)
        
        # Add hydrogen if it exists
        if hydrogen_count > 0:
            hill_notation += 'H'
            if hydrogen_count > 1:
                hill_notation += str(hydrogen_count)
        
        # Add other elements
        for elem, count in sorted_other_elements:
            hill_notation += elem
            if count > 1:
                hill_notation += str(count)
        
        return hill_notation

    def get_mass(self):
        mass = 0
        periodic_table = Chem.GetPeriodicTable()
        for atom, value in self.dict_representation.items():
            try:
                atom_mass = periodic_table.GetMostCommonIsotopeMass(atom)
            except RuntimeError:
                return None
            mass += atom_mass * value
        return mass