(ns util.regularization)

;------- regularization

(defn l2-regularization
  "Returns the L2 regularization function f [w] that rescales its argument w to:

      w * (1.0 - (eta * lambda)

  Where eta is the learning rate and lambda is the regularization parameter."
  [eta lambda]
  (let [rescale (- 1.0 (* eta lambda))]
    (fn [weight]
      (* weight rescale))))

(defn l1-regularization
  "Returns the L1 regularization function f [w] that rescales its argument w to:

      w - (eta * lambda) if w < 0.0
      w + (eta * lambda) if w >= 0.0

  Where eta is the learning rate and lambda is the regularization parameter."
  [eta lambda m]
  (let [rescale (* eta lambda)]
    (fn [weight]
      (if (< weight 0.0)
        (- weight rescale)
        (+ weight rescale)))))
