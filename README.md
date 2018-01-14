# detect-gomoku-using-neural-network
This project is used to detect X, O or white block in a gomoku player board. It use neural network to detect 3 image above. Then it draw on computer exactly the the element in each blocks.
## Training
In here, I use 3 training data set that I collect myself using a webcame. I use a board 10x10 and draw X, O respectively, then take picture in each block and save data. White picture, I use a filter to remove all black strokes in X and O with threshold is 230.
## Running the test
1. Training using training_vs3.py.
2. Testing using project.py. Remember adjust 2 variable: real_col and real_row to exact row and column of chess board you had drawn/printed
