function Matrix = padding(data)
[rows, cols, bands]  = size(data);
Matrix = zeros(rows+2, cols+2, bands);
Matrix(2:end-1, 2:end-1, :) = data;
Matrix(1, 2:end-1, :) = data(1,:,:); % up
Matrix(end, 2:end-1, :) = data(end,:,:); % down
Matrix(2:end-1, 1, :) = data(:,1,:); % left
Matrix(2:end-1, end, :) = data(:,end,:); % right
Matrix(1, 1, :) = Matrix(1,2,:); % upper left
Matrix(1, end, :) = Matrix(1,end-1,:); % upper right
Matrix(end, 1, :) = Matrix(end,2,:); % lower left
Matrix(end, end, :) = Matrix(end,end-1,:); % lower right
end