import glob

def assign_correct_jpg(path):
    all_files = glob.glob(path)
    for ff in all_files:
        with open(ff) as f:
            lines = f.readlines()
        for ll in lines:
            if "guid:" in ll:
                guid_ff = ll.strip()

        with open(ff.replace('.jpg.meta', '.mat')) as f:
            lines = f.readlines()
        for ii, ll in enumerate(lines):
            if 'm_Texture: {fileID: 0}' in ll:
            # if "m_Texture: {fileID: 2800000, " in ll:
                lines[ii] = "        m_Texture: {fileID: 2800000, " + guid_ff + ", type: 3}"
        
        f = open(ff.replace('.jpg.meta', '.mat'), "w")
        f.write(''.join(lines))
        f.close()

assign_correct_jpg("Assets/Resources/Textures/Multi-Garmentdataset/*.jpg.meta")
assign_correct_jpg("Assets/Resources/Textures/people_snapshot_public/*.jpg.meta")
assign_correct_jpg("Assets/Resources/Textures/SURREAL/female/*.jpg.meta")
assign_correct_jpg("Assets/Resources/Textures/SURREAL/male/*.jpg.meta")
