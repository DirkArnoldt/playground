(ns util.matrix)

(defn to-matrix [n coll]
  (vec (map vec (partition n coll))))

(defn initialize-matrix [[i j] v]
  (to-matrix j (repeat (* i j) v)))

(defn initialize-matrix2 [[i j] f]
  (to-matrix j (repeatedly (* i j) f)))

(defn row-count [m]
  (count m))

(defn column-count [m]
  (let [[col & _] m]
    (count col)))

(defn dimension [m]
  (let [si (row-count m)
        sj (column-count m)]
    (vector si sj)))

(defn row [m i]
  (m i))

(defn column [m i]
  (vec (map #(nth % i) m)))

(defn columns [m]
  (vec
    (for [c (range (column-count m))]
      (column m c))))

(defn transpose [m]
  (apply vector (columns m)))

(defn build-indexes [[si sj]]
  (for [i (range si) j (range sj)]
    [i j]))

(defn get-indexes [m]
  (build-indexes (dimension m)))

(defn etransform [f m idx]
  (assoc-in m idx (f (get-in m idx))))

(defn etransform-with-index [f m idx]
  (assoc-in m idx (f (get-in m idx) idx)))

(defn- mtransform [f m]
  (reduce f m (get-indexes m)))

(defn transform [f m]
  (mtransform (partial etransform f) m))

(defn transform-with-index [f m]
  (mtransform (partial etransform-with-index f) m))

(defn habamard [a b]
  (mapv * a b))

(defn dotproduct [a b]
  (reduce + 0.0 (habamard a b)))

(defn mapm [f a b]
  (to-matrix (column-count a) (mapv f (flatten a) (flatten b))))