function subimg_n=nor(subimg)
subimg_n= (subimg-min(subimg(:)))/(max(subimg(:))-min(subimg(:)));