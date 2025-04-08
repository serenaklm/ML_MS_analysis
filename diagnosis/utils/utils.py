import json
import pickle
import rdkit.Chem as Chem

def load_pickle(path):

    with open(path, "rb") as f:

        data = pickle.load(f)
    
    return data

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def pickle_data(data, path):

    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_mol(smiles):

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx() + 1))
    
    return mol


def update_dict(entry, rec, key):

    rec[key] = entry

    return rec
