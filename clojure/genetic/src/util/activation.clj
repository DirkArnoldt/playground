(ns util.activation)

(defn sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- 0 x)))))

(defn sigmoid' [x]
  (let [sigx (sigmoid x)]
    (* sigx (- 1 sigx))))

(defn softplus [x]
  (Math/log (+ 1.0 (Math/exp x))))

(def softplus' sigmoid)

(defn tanh [x]
  (Math/tanh x))

(defn tanh' [x]
  (let [tanhx (Math/tanh x)]
    (* (+ 1 tanhx) (- 1 tanhx))))

