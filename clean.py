import os
import shutil

def data_and_fig():
    folder_names = ['data', 'fig']
    for item in folder_names:
        shutil.rmtree(item, ignore_errors=True)
        os.mkdir(item)

def directory(dirname):
    shutil.rmtree(dirname, ignore_errors=True)
    os.mkdir(dirname)
        
def these_extensions(file_extensions):
    all_files = os.listdir(os.getcwd())
    for file in all_files:
        for ext in file_extensions:
            if file.endswith(ext):
                remove_if_exists(file)
def remove_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)
        return(print("Removed", filename))
    else:
        pass

if __name__ == '__main__':
    pass
    # print("Cleaning..")
    # yesorno = 'yes'
    # #yesorno = input("Do you want to clean the data ? (yes or no)\n")
    # if yesorno =='yes':
    #     data_and_fig()
    #     these_extensions(['.aux', '.log', '.gz', '.dvi', '.ps'])
    #     print("Done")
    # else:
    #     print("Do nothing")
    