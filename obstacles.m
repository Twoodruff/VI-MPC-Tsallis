function obs = obstacles(origin,box_size,num_obs)
%OBSTACLES Creates array of evenly spaced square obstacles in 2D grid.
%   Detailed explanation goes here

% Create points field (edges of all obstacles)
points = zeros(2*num_obs,2*num_obs,2);

for i=1:size(points,2)
    for j=1:size(points,1)
        points(i,j,1) = origin(1) + (i-1)*box_size;
        points(i,j,2) = origin(2) + (j-1)*box_size;
    end
end

% Create obstacle field using polyshapes
poly = [];
for i=1:2:2*num_obs
    for j=1:2:2*num_obs
        xp = [points(i,j,1),points(i+1,j,1),points(i+1,j+1,1),points(i,j+1,1)];
        yp = [points(i,j,2),points(i+1,j,2),points(i+1,j+1,2),points(i,j+1,2)];
        poly = [poly polyshape(xp,yp)];
    end
end

obs = union(poly);

end

