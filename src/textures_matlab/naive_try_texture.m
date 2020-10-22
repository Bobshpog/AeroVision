I = imread("circles2.png");
map = hsv(256);
rgbImage = ind2rgb(I, map); 
A = [
    9.99963954e-01  -8.49057398e-03 0 -4.74510255e-02;
    1.61827673e-04 1.90589989e-02 9.99818348e-01 -2.57221580e-02;
    -8.48903165e-03 -9.99782309e-01 1.90596859e-02 -5.34192476e-02;
    0 0 0 1];
B = inv(A);
C = zeros(824,519);
tform = affine3d(A.');
J = imwarp(rgbImage,tform);
for v= 1:20 %gave a small portion so tried combination
    s = size(J);
    k = reshape(J(9,:,:),[s(2),s(3)]);

    C= C+ k;
end
figure
imshow(k)

figure
imshow(C)
impixelinfo();

