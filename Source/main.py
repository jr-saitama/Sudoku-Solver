import os
# path = os.getcwd()
# os.chdir(os.path.join(path,"Source"))
from grid_parser import get_cells
from Solver import display, solve
from number_predict import predict

def start_solve():
	get_cells("sudoku.jpg")
	predicted_str = predict("temp_dir_for_cell/square")
	predicted_str = "003042090090060500500000010001700285008000100329008700030000001005090020080210600"
	print(predicted_str)
	display(solve(predicted_str))





if __name__ == '__main__':
	start_solve()
