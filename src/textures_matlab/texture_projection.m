I = imread("rainbow_wing.png");
im = imread("rainbow.png");
M = [76 379;
    150 379;
    151 302;
    77 303;
    
    303 303;
    303 229;
    347 298;
    349 304
    ];
N = [814 709;
    856 622;
    969 631;
    993 721;
    
    946 564;
    1011 570;
    993 552;
    945 549
    ];
figure(1)   
imshow(I);
figure(2)
imshow(im);
sz = size(I);
%outputView = imref2d([30000,10000,3]); % use this with invert, real output
%is much bigger
outputView = imref2d(sz);
tform = estimateGeometricTransform(M,N,'projective');
%tform = invert(tform);
Ir = imwarp(im,tform,'OutputView',outputView);
figure(3) 
imshowpair(I,Ir,'montage');




function geoTry()
    original  = imread('rainbow.png');
    distorted  = imread('rainbow_wing.png');
    gray_rainbow = rgb2gray(original);
    size(distorted);
    g_distorted = rgb2gray(distorted);
    ptsOriginal  = detectSURFFeatures(gray_rainbow);
    ptsDistorted = detectSURFFeatures(g_distorted);
    [featuresOriginal,validPtsOriginal] = extractFeatures(gray_rainbow,ptsOriginal);
    [featuresDistorted,validPtsDistorted] = extractFeatures(g_distorted,ptsDistorted);
    index_pairs = matchFeatures(featuresOriginal,featuresDistorted);
    matchedPtsOriginal  = validPtsOriginal(index_pairs(:,1));
    matchedPtsDistorted = validPtsDistorted(index_pairs(:,2));
    figure 
    showMatchedFeatures(gray_rainbow,g_distorted,matchedPtsOriginal,matchedPtsDistorted);

    [tform,inlierIdx] = estimateGeometricTransform(matchedPtsDistorted,matchedPtsOriginal,'similarity');
    inlierPtsDistorted = matchedPtsDistorted(inlierIdx,:);
    inlierPtsOriginal  = matchedPtsOriginal(inlierIdx,:);

    figure 
    showMatchedFeatures(original,distorted,inlierPtsOriginal,inlierPtsDistorted)
    outputView = imref2d(size(original));
    Ir = imwarp(distorted,tform,'OutputView',outputView);
    figure 
    imshow(Ir); 
end
