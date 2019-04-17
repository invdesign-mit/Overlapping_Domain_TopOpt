function epsilon=mat_silicon(lamnm)

lam=lamnm/1000;
epsilon=11.67316 + 1/lam^2 + 0.004482633/(lam^2 - 1.108205^2);

end