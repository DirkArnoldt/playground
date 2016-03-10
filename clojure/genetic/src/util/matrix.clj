(ns util.matrix)

(defn to-matrix [n coll]
  (vec (map vec (partition n coll))))

(defn initialize-matrix [[i j] v]
  (to-matrix j (repeat (* i j) v)))

(defn initialize-matrix2 [[i j] f]
  (to-matrix j (repeatedly (* i j) f)))

(defn dimension [m]
  (let [[first & _] m
        si (count m)
        sj (count first)]
    (vector si sj)))

(defn row [m i]
  (m i))

(defn column [m i]
  (map #(nth % i) m))

(defn build-indexes [[si sj]]
  (for [i (range si) j (range sj)]
    [i j]))

(defn get-indexes [m]
  (build-indexes (dimension m)))

(defn etransform [f m idx]
  (assoc-in m idx (f (get-in m idx))))

(defn etransform-with-index [f m idx]
  (assoc-in m idx (f (get-in m idx) idx)))

(defn transform [f m]
  (let [transformer (partial etransform f)]
    (reduce transformer m (get-indexes m))))

(defn transform-with-index [f m]
  (let [transformer (partial etransform-with-index f)]
    (reduce transformer m (get-indexes m))))
