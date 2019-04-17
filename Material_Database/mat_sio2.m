function epsilon=mat_sio2(lamnm)

lam=lamnm/1000;
epsilon=1 + 0.6961663*lam^2/(lam^2-0.0684043^2) + 0.4079426*lam^2/(lam^2-0.1162414^2) + 0.8974794*lam^2/(lam^2-9.896161^2);

end