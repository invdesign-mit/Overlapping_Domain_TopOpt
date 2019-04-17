function epsilon=mat_alox(lamnm)

lam=lamnm/1000;
file='alox_refractiveindex_[nm].txt';
data=load(file);

epsilon=interp1(data(:,1),data(:,2),lam)^2;

end