import matplotlib.pyplot as plt
import importrhdutilities as rhd

def generate_frame(data):
    plt.imshow( data[:,0:100] )
    plt.show()
    

def read_file(path):
    result, data_present = rhd.load_file(path)
    if data_present:
        # print(result['amplifier_data'])
        print("Total",result['amplifier_data'].shape[0],"channels,",result['amplifier_data'].shape[1],"samples.")
        
        return result
    else:
        print('Plotting not possible; no data in this file')
    

if __name__ == '__main__':
    path = "rawdata/dorsi-plantar_170306_124618.rhd"
    ret = read_file(path)
    print(ret['amplifier_data'][0].shape)
    generate_frame(ret['amplifier_data'])
    

