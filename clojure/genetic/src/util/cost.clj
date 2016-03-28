(ns util.cost)

(defn error [a b]
  (- a b))

(defn squared-error [a b]
  (let [e (error a b)]
    (* e e)))

(defn log-likelihood [a b]
  (let [res (+ (* b (Math/log a))
               (* (- 1.0 b) (Math/log (- 1.0 a))))]
    (if (Double/isNaN res)
      0.0
      res)))




