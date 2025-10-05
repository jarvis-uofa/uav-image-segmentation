import pandas as pd 


def main () : 
    csv = "rgb_interpolated_corners.csv"
    csv_mod = "rgb_inter_mod.csv"
    column_to_rm = ["top_left_x","top_left_y", "bottom_right_x", "bottom_right_y"]
    temp = drop_columns(csv, column_to_rm) 
    temp.to_csv(csv_mod, index=False) 
    return 0
    
def drop_columns(csv_name, colmnrm) : 
    dataframe = pd.read_csv(csv_name) 
    dataframe = dataframe.drop(columns = colmnrm) 
    return dataframe 

if __name__ == '__main__' :
    print(main()) 
    