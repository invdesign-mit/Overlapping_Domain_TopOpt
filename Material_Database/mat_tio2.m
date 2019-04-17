function epsilon=mat_tio2(lam)

file='tio2_refractiveindex_[nm].txt';
data=load(file);

epsilon=interp1(data(:,1),data(:,2),lam)^2;

end