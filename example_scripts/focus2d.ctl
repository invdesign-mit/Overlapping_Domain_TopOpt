(define-param Nx 10000)
(define-param Ny 249)
(define-param Npmlx 0)
(define-param Npmly 25)
(define-param hx 0.02)
(define-param hy 0.02)
(define-param iy_src 27)
(define-param iy_mtr 222)
(define-param fcen 1.0)
(define-param theta 15)
(define-param inputfilename "spec1_epsilon.h5")
(define-param focal_length 100)
(define-param farfield_x 26.79491924311227)
(define-param farfield_xsize 50)
(define-param farfield_ysize 0.1)
(define-param farfield_dh 0.1)
(define-param flux0? false)
(define-param printff? true)
(define-param nsub 1.53)

(define-param Nxo 0)
(define-param df 0.1)
(define-param res (/ 1 hx))
(define-param comp Ez)

(define-param thetarad  (/ (* theta pi) 180) )
(define-param sx (* Nx hx))
(define-param sy (* Ny hy))
(define-param dpmlx (* Npmlx hx))
(define-param dpmly (* Npmly hy))

(define-param jy0 (- (* iy_src hy) (/ sy 2)))
(define-param ry0 (- (* iy_mtr hy) (/ sy 2)))
(define-param rsx (- sx (* 2 Nxo hx)))

(reset-meep)
(set-param! resolution res)
(set! geometry-lattice (make lattice (size sx sy no-size)))
(set! pml-layers (list 
      		 (make pml (direction X) (thickness dpmlx))
      		 (make pml (direction Y) (thickness dpmly))
))
(define-param k0 (* 2 pi fcen))
(define-param kx (* k0 nsub (sin thetarad)))
(set! k-point (vector3 (/ kx (* 2 pi)) 0 0))

(set! force-complex-fields? true)
(set! eps-averaging? false)

(if flux0?
    (set! default-material (make dielectric (epsilon (* nsub nsub))))
    (set-param! epsilon-input-file inputfilename)
)

(define (my-amp-func p) (exp (* 0+1i nsub k0 (sin thetarad) (vector3-x p))))

(set! sources (append (list
              	      (make source (src (make gaussian-src (frequency fcen) (fwidth df)))
                            	   (component comp) 
				   (center 0 jy0 0)
				   (size sx 0 0)
				   (amp-func my-amp-func))
)))

(define nearfield 
	(add-near2far fcen 0 1 
		      (make near2far-region (center 0 ry0 0) (size rsx 0 0))
))

(define-param nfreq 50) 
(define trans (add-flux fcen df nfreq
	      (make flux-region (center 0 ry0 0) (size rsx 0 0))
))  

(run-sources+ (stop-when-fields-decayed (/ 10 fcen) comp (vector3 0 ry0 0) 1e-5)
	      ;(at-beginning output-epsilon)
)

(display-fluxes trans)

(define-param ffx farfield_x)
(define-param ffy (+ ry0 focal_length))
(define-param ffLx farfield_xsize)
(define-param ffLy farfield_ysize)
(define-param ffdL farfield_dh)
(define-param ffres (/ 1 ffdL))

(if printff?
(output-farfields nearfield
 (string-append "farfield-freq" (number->string fcen) "-angle" (number->string theta))
 (volume (center ffx ffy 0) (size ffLx ffLy))
 ffres)
)

