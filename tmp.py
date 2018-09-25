import os  , glob

paths = glob.glob('/Users/seongjungkim/PycharmProjects/Find_Wally/background/cropped_bg/*')
for i,path in enumerate(paths):
    root_dir , name =os.path.split(path)
    name=os.path.splitext(root_dir)[0]
    os.rename(path ,  os.path.join(root_dir , str(i) +'.jpg'))

