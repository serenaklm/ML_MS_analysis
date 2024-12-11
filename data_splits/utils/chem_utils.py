import rdkit.Chem as Chem
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

def get_mol(smiles):

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx() + 1))
    
    return mol

def get_scaffold(smiles):

    mol = get_mol(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    return Chem.MolToSmiles(scaffold)

def scaffold_split(data, test_size = 0.2, random_state = 17):

    smiles = [r["smiles"] for r in data]
    scaffolds = [get_scaffold(s) for s in smiles]
    
    train_scaffolds, test_scaffolds = train_test_split(scaffolds, test_size = test_size, random_state = random_state)
    train_scaffolds, val_scaffolds = train_test_split(train_scaffolds, test_size = test_size, random_state = random_state)

    train_idx = [i for i, s in enumerate(scaffolds) if s in train_scaffolds]
    val_idx = [i for i, s in enumerate(scaffolds) if s in val_scaffolds]
    test_idx = [i for i, s in enumerate(scaffolds) if s in test_scaffolds]

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    return train_data, val_data, test_data
