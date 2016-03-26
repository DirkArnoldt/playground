(ns util.activation)

;-------- activation functions

(defn sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- 0.0 x)))))

(defn sigmoid' [x]
  (let [sigx (sigmoid x)]
    (* sigx (- 1.0 sigx))))

(defn softplus [x]
  (Math/log (+ 1.0 (Math/exp x))))

(def softplus' sigmoid)

(defn tanh [x]
  (Math/tanh x))

(defn tanh' [x]
  (let [tanhx (Math/tanh x)]
    (* (+ 1.0 tanhx) (- 1.0 tanhx))))


;------------- cost functions

(defn quadratic-cost [target output]
  (* 0.5 (reduce + 0.0 (mapv #(* (- %1 %2) (- %1 %2)) target output))))

(defn cross-entropy-cost [target output]
  (reduce + (mapv (fn [y a]
                    (let [res (- (* (* -1.0 y)
                                    (Math/log a))
                                 (* (- 1.0 y)
                                    (Math/log (- 1.0 a))))]
                      (if (Double/isNaN res)
                        0.0
                        res))) target output)))


;------- regularization

(defn l2-regularization
  "Returns the L2 regularization function f [w] that rescales its argument w to:

      (1.0 - (eta * lambda / m) * w

  Where eta is the learning rate, lambda is the regularization parameter and m denotes the
  number of training samples."
  [eta lambda m]
  (let [rescale (- 1.0 (/ (* eta lambda) m))]
    (fn [weight]
      (* weight rescale))))

(defn l1-regularization
  "Returns the L1 regularization function f [w] that rescales its argument w to:

      w - (eta * lambda / m) if w < 0.0
      w + (eta * lambda / m) if w >= 0.0

  Where eta is the learning rate, lambda is the regularization parameter and m denotes the
  number of training samples."
  [eta lambda m]
  (let [rescale (/ (* eta lambda) m)]
    (fn [weight]
      (if (< weight 0.0)
        (- weight rescale)
        (+ weight rescale)))))


;--------- random with gaussian distribution

(def generator (java.util.Random.))

(defn rand-gaussian
  ([] (rand-gaussian 0 1))
  ([mean standard-deviation]
   (-> (.nextGaussian generator)
       (* standard-deviation)
       (+ mean))))


(defn randit [i]
  (fn [] (rand-gaussian 0 (/ 1.0 (Math/sqrt i)))))
