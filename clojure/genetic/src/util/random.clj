(ns util.random)

;--------- random with gaussian distribution

(def generator (java.util.Random.))

(defn rand-gaussian
  ([] (rand-gaussian 0 1))
  ([mean standard-deviation]
   (-> (.nextGaussian generator)
       (* standard-deviation)
       (+ mean))))

(defn rand-for [i]
  (fn [] (rand-gaussian 0 (/ 1.0 (Math/sqrt i)))))
