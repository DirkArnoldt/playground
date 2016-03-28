(ns util.matrix)

(defn to-matrix [n coll]
  (vec (map vec (partition n coll))))

(defn compute-matrix [[i j] f]
  (to-matrix j (repeatedly (* i j) f)))

(def zero-matrix (memoize
                   (fn [dim]
                     (compute-matrix dim (constantly 0.0)))))

(defn compute-vector [i f]
  (vec (repeatedly i f)))

(def zero-vector (memoize
                   (fn [i]
                     (compute-vector i (constantly 0.0)))))

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

(defn dot [a b]
  (reduce + 0.0 (habamard a b)))

(defn mapm [f a b]
  (mapv (fn [ea eb]
          (if (coll? ea)
            (mapm f ea eb)
            (f ea eb))) a b))

; vector
(defn add [a b]
  (mapv + a b))

(defn sub [a b]
  (mapv - a b))