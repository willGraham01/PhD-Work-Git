# Script preserves mesh files for use in plotting after multiple runs over various values of N.
# See Store_Mesh_Files for documentation details.

from shutil import copyfile

def DeduceN(gFile, line=11):
    '''
    Deduce value of N from the .geo file gFile.
    Optionally, provide the index of the line where this definition appears,
    default search will be on line 12 (index 11) as per the established convention.
    '''
    with open('FEM_CrossGraphSetup.geo') as fp:
        for i, line in enumerate(fp):
            if i == 11:
                # 11th line
                Nstr = line[4:-2] #take all characters after the numerical value appears in the file
            elif i > 11:
                #terminate loop early
                break
    # We want N as a float because we will use it in computations, to define points on meshes etc.
    # Int might cause bugs.
    for i, char in enumerate(Nstr):
        if char==';':
            Nstr = Nstr[:i]
            break
    N = float(Nstr)
    return N, Nstr

if __name__=="__main__":
    # This is the name of our mesh files, minus the extension.
    filePrefix = 'FEM_CrossGraphSetup'
    # This is the gmsh file which generated our domain's mesh. We can infer information about our domain from it
    gmshFile = filePrefix + '.geo'
    meshFolder = 'MeshDump'
    meshCreated = filePrefix + '.xml'

    Nstr = DeduceN(gmshFile)[1]
    meshSave = './' + meshFolder + '/Mesh-N' + Nstr + '.xml'

    copyfile(meshCreated, meshSave)
