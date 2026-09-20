r"""
Abelian surfaces

This library extends existing functionality for computing with
- Jacobians of genus-2 hyperelliptic curves,
- products of two elliptic curves, and
- quaternion matrices under the Ibukiyama--Katsuri--Oort (IKO)
  correspondence.

In addition, we implement formulae for polarised `(2,2)`-isogenies between
superspecial principally polarised abelian surfaces, as well as the KLPT`^2`
algorithm (TODO).

Sources:
- [Kunzweiler24] (https://ia.cr/2022/990) (Richelot isogenies)
- [OudomphengPope22] (https://ia.cr/2022/1283) (gluing isogenies)
- [CDKLPT25] (https://ia.cr/2025/372) (IKO Correspondence)

AUTHORS:

- Jamie Low Jiaqi (2026)
"""

# To be organised
from collections.abc import Callable, Iterable
from functools import cached_property
from itertools import combinations
from sage.algebras.quatalg.quaternion_algebra import QuaternionAlgebra, QuaternionAlgebra_ab, QuaternionOrder
from sage.algebras.quatalg.quaternion_algebra_element import QuaternionAlgebraElement_rational_field
from sage.arith.misc import gcd
from sage.categories.commutative_additive_groups import CommutativeAdditiveGroups
from sage.categories.schemes import Jacobians
from sage.groups.perm_gps.permgroup_named import SymmetricGroup
from sage.matrix.constructor import matrix, Matrix
from sage.matrix.matrix_generic_dense import Matrix_generic_dense
from sage.matrix.matrix_integer_dense import Matrix_integer_dense
from sage.matrix.matrix_rational_dense import Matrix_rational_dense
from sage.matrix.matrix_space import MatrixSpace
from sage.misc.cachefunc import cached_method
from sage.misc.functional import sqrt
from sage.modules.free_module import VectorSpace
from sage.modules.free_module_element import vector
from sage.rings.finite_rings.element_base import FiniteRingElement
from sage.rings.finite_rings.finite_field_base import FiniteField
from sage.rings.finite_rings.finite_field_constructor import GF
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring import polygens, polygen
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.polynomial.polynomial_zz_pex import Polynomial_ZZ_pEX
from sage.rings.rational import Rational
from sage.rings.rational_field import QQ
from sage.schemes.elliptic_curves.constructor import EllipticCurve_from_cubic, EllipticCurve
from sage.schemes.elliptic_curves.ell_finite_field import EllipticCurve_finite_field
from sage.schemes.elliptic_curves.ell_point import EllipticCurvePoint_finite_field
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
from sage.schemes.hyperelliptic_curves.hyperelliptic_g2 import HyperellipticCurve_g2_finite_field
from sage.schemes.hyperelliptic_curves.jacobian_g2_generic import HyperellipticJacobian_g2_generic
from sage.schemes.hyperelliptic_curves.jacobian_morphism import MumfordDivisorClassField
from sage.schemes.weighted_projective.weighted_projective_point import SchemeMorphism_point_weighted_projective_ring as ProjectivePoint
from sage.sets.cartesian_product import CartesianProduct
from sage.structure.sage_object import SageObject
from sage.modules.vector_integer_dense import Vector_integer_dense
from sage.modules.vector_rational_dense import Vector_rational_dense
from typing import Any


def sum(data: Iterable):
    """
    Return the sum of the elements of ``data``.
    """
    result = 0
    for p in data:
        if not result:
            result = p
        else:
            result += p
    return result


def dot(v1: Iterable, v2: Iterable) -> Any:
    """
    Return the dot product of ``v1`` and ``v2``.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length.")
    return sum(v1[i] * v2[i] for i in range(len(v1)))


class HyperellipticJacobian_g2_generic(HyperellipticJacobian_g2_generic):
    r"""
    The Jacobian of a genus-2 hyperelliptic curve
    `\mathcal{C}/\mathbb{F}_q:y^2=f(x)`.

    INPUT:

    - ``curve`` -- either the curve `\mathcal{C}` or the polynomial `f(x)`.

    - ``custom_name`` -- (default: None) a custom string representation for
      this object.

    - ``roots`` -- (default: None) the roots of `f(x)`. If specified,
      computations involving the roots will use this datum, avoiding
      polynomial factorisations.

    - ``type_1_inv`` -- (default: None) the type-1 invariants
      `(A,B,C,E)` of `\mathcal{C}` as in [Kunzweiler24].

    - ``type_2_inv`` -- (default: None) the type-2 invariants
      `(A,B,C,E)` of `\mathcal{C}` as in [Kunzweiler24].

    - ``*args`` -- additional arguments to pass to the
      ``HyperellipticJacobian_g2_generic`` constructor.

    - ``**kwargs`` -- additional keyword arguments to pass to the
      ``HyperellipticJacobian_g2_generic`` constructor.
    """

    def __init__(self,
                 curve: HyperellipticCurve_g2_finite_field | Polynomial_ZZ_pEX,
                 **kwargs):
        r"""
        Create the Jacobian of a genus-2 hyperelliptic curve
        `\mathcal{C}/\mathbb{F}_q:y^2=f(x)`.

        INPUT:

        - ``curve`` -- either the curve `\mathcal{C}` or the polynomial `f(x)`.

        - ``custom_name`` -- (default: None) a custom string representation for
          this object.

        - ``roots`` -- (default: None) the roots of `f(x)`. If specified,
          computations involving the roots will use this datum, avoiding
          polynomial factorisations.

        - ``type_1_inv`` -- (default: None) the type-1 invariants
          `(A,B,C,E)` of `\mathcal{C}` as in [Kunzweiler24].

        - ``type_2_inv`` -- (default: None) the type-2 invariants
          `(A,B,C,E)` of `\mathcal{C}` as in [Kunzweiler24].
        """

        if isinstance(curve, HyperellipticCurve_g2_finite_field):
            self._curve: HyperellipticCurve_g2_finite_field = curve
            f = curve.hyperelliptic_polynomials()[0]
            self._poly: Polynomial_ZZ_pEX = f
        elif isinstance(curve, Polynomial_ZZ_pEX):
            C = HyperellipticCurve(curve)
            self._curve: HyperellipticCurve_g2_finite_field = C
            self._poly: Polynomial_ZZ_pEX = curve
        else:
            raise ValueError(
                "curve should be of type HyperellipticCurve_g2_finite_field or"
                " Polynomial_ZZ_pEX."
            )

        F = self._curve.base_ring()
        super().__init__(self._curve, category=Jacobians(F))

        self.type_1_inv = kwargs.get('type_1_inv')
        self.type_2_inv = kwargs.get('type_2_inv')
        self._roots = kwargs.get('roots')
        if custom_name := kwargs.get('custom_name'):
            self.rename(custom_name)

        Fx = PolynomialRing(F, 'x')
        self._F = F
        self._Fx = F, Fx, Fx.gen()

    @cached_method
    def curve(self) -> HyperellipticCurve_g2_finite_field:
        """
        Return the underlying genus-2 hyperelliptic curve.
        """
        return self._curve

    @cached_method
    def polynomial(self) -> Polynomial_ZZ_pEX:
        """
        Return the underlying hyperelliptic polynomial.
        """
        return self._poly

    poly = polynomial

    @cached_method
    def zero(self) -> MumfordDivisorClassField:
        r"""
        Return the identity of the underlying abelian group.

        Alias of ``self()``.
        """
        return self()

    @cached_method
    def base_ring(self) -> FiniteField:
        r"""
        Return the finite field `\mathbb{F}_q` which ``self`` is defined over.
        """
        return self._curve.base_ring()

    base_field = base_ring

    def change_ring(self, K: FiniteField, **kwargs) \
            -> HyperellipticCurve_g2_finite_field:
        """
        Return the base change of ``self`` to the field `K`.

        INPUT:

        - ``**kwargs`` -- additional keyword arguments to pass to the
          ``HyperellipticJacobian_g2_generic`` constructor.
        """
        return HyperellipticJacobian_g2_generic(self._curve.change_ring(K),
                                                **kwargs)

    base_extend = change_ring

    @cached_method
    def leading_coefficient(self) -> FiniteRingElement:
        """
        Return the leading coefficient of `f(x)`.
        """
        return self._poly.leading_coefficient()

    lc = leading_coefficient

    def __call__(self, *args: tuple) -> MumfordDivisorClassField:
        r"""
        Return a point of ``self`` (i.e., divisor class of `\mathcal{C}'),
        specified by ``*args``.

        INPUT:

        - ``args`` can take one of several forms:
            - ``()`` -- an empty tuple.
            - ``(u, v)`` -- a pair of reduced Mumford divisors.
            - ``(u, v, n)`` -- a pair of reduced Mumford divisors with weight
              `0 \leq n \leq 2-\deg(u)`.
            - ``(P,)`` -- a point on `\mathcal{C}`.
            - ``(P, n)`` -- a point on `\mathcal{C}` and an integer
              `n\in\{0,1,2\}`.
            - ``(P, Q)`` -- a pair of points on `\mathcal{C}`.

          Each point `(X:Y:Z)\in\mathcal{C}` can be specified as:
            - a ``ProjectivePoint`` object.
            - its affine coordinates ``(X, Y)``, if `Z=1`.
            - its projective coordinates ``(X, Y, Z)``.

        OUTPUT:

        - If ``args`` is ``()``, return the principal divisor class.
        - If ``args`` is ``(u, v)``, return the divisor class with Mumford
          coordinates `(u, v)`.
        - If ``args`` is ``(u, v, n)``, return the divisor class with Mumford
          coordinates `(u, v : n)`.
        - If ``args`` is ``(P,)``, return the divisor class `[P-\infty]` (if
          `f(x)` has degree 5) or `[P-\infty_-]` (if `f(x)` has degree 6).
        - If ``args`` is ``(P, n)`` where ``n`` is 0 or 1 and `f(x)` has degree
          6, return the divisor class `[P-\infty_+]` (if `n=0`) or
          `[P-\infty_-]` (if `n=1`).
        - If ``args`` is ``(P, 2)``, return the divisor class `[2P-2\infty]`
          (if `f(x)` has degree 5) or `[2P-\infty_+-\infty_-]` (if `f(x)` has
          degree 6).
        - If ``args`` is ``(P, Q)``, return the divisor class `[P-Q]`.

        EXAMPLES::

            sage: F.<ω> = GF((6143, 2), 'ω', x^2 + 1)
            sage: x = polygen(F)
            sage: A1 = HyperellipticJacobian_g2_generic(x^5 + 1); A1
            Jacobian of Hyperelliptic Curve over Finite Field in ω of size
            6143^2 defined by y^2 = x^5 + 1
            sage: A1(x^2 + 6142*x, 2645*x + 6142)
            (x^2 + 6142*x, 2645*x + 6142)
            sage: A1((0, 1), (1, 2644))
            (x^2 + 6142*x, 2645*x + 6142)
        """
        if (not args or args[:2] == (1, 0)
                or hasattr(args[0], "scheme")
                or hasattr(args[0], "splitting_field")):

            if not isinstance(args[0], ProjectivePoint):
                # Mumford divisor
                if self._poly.degree() == 5:
                    # input: (u, v) or (u, v, n); ignore n
                    return super().__call__(*args[:2])
                if len(args) == 3:
                    # input: (u, v, n)
                    return super().__call__(*args)
                # input: (u, v); set n to default (n = 2 – deg u)
                try:
                    d = args[0].degree()
                except:
                    d = 0
                return super().__call__(args[0], args[1], 2 - d)
                    
            if len(args) == 1:
                # input: (P,)
                if self._poly.degree() == 5:
                    return super().__call__(args[0])
                elif args[0][2] != 0:
                    # affine point
                    args += (1,)
                else:
                    # P = ∞-
                    args += (2,)

            _, _, x = self._Fx
            if not isinstance(args[1], ProjectivePoint):
                # input: (P, n)
                P, n = args
                if P[2] == 0:
                    # point at infinity
                    w = 1 if P[1] < -P[1] else 0
                    # P = ∞+ if w == 1 else P = ∞-
                    return super().__call__(
                        1, 0,
                        0 if (w, n) in [(0, 0), (0, 2)] else  # ∞- – ∞+
                        1 if (w, n) in [(0, 1), (1, 0)] else  # 0
                        2 if (w, n) in [(1, 1), (1, 2)] else  # ∞+ – ∞-
                        None
                    )
                u, v = x - P[0], P[1]
                if n < 2:
                    # input: (P, 0) or (P, 1)
                    return self(u, v, n)
                else:
                    # input: (P, 2)
                    return self(u, v, 0) + self(u, v, 1)

            # input: (P, Q)
            P, Q = args
            if self._poly.degree() == 5:
                return super().__call__(*args)
            # degree 6
            if P[2] == 0:
                # P is a point at infinity
                wP = 1 if P[1] < -P[1] else 0
                if Q[2] == 0:
                    # Q is a point at infinity
                    wQ = 1 if Q[1] < -Q[1] else 0
                    return super().__call__(
                        1, 0,
                        0 if (wP, wQ) == (0, 1) else  # ∞- – ∞+
                        2 if (wP, wQ) == (1, 0) else  # ∞+ – ∞-
                        1                             # 0
                    )
                # -[Q – P] where Q ≠ ∞± and P = ∞±
                u, v = x - Q[0], Q[1]
                return -super().__call__(u, v, 1 - wP)
            elif Q[2] == 0:
                # [P – Q] where P ≠ ∞± and Q = ∞±
                wQ = 1 if Q[1] < -Q[1] else 0
                u, v = x - P[0], P[1]
                return super().__call__(u, v, 1 - wQ)
            return super().__call__(*args)

        if not isinstance(args[0], list | tuple):
            # 1 set of coordinates (x:y:z)
            args = [args]
        assert len(args) <= 2, (
            "At most two sets of coordinates can be provided."
        )
        # TODO: extend field when necessary
        points = [self._curve(*c) for c in args]
        return self(*points)

    @cached_method
    def points_of_order_2(self) -> tuple[MumfordDivisorClassField]:
        """
        Return the 15 points of order 2 of ``self``.
        """
        return (self(P, Q) for P, Q in
                combinations(self.weierstrass_points(), 2))

    @cached_method
    def roots(self) -> tuple[FiniteRingElement]:
        r"""
        Return the roots of ``self``.

        OUTPUT:

        - If `f(x)` has degree 5, return a tuple containing its 5 roots
          over `\overline{\mathbb{F}}_q`, followed by ``None``
          (representing `\infty`).
        - If `f(x)` has degree 6, return a tuple containing its 6 roots
          over `\overline{\mathbb{F}}_q`.
        """
        if self._roots is None:
            factorisation = self._poly.factor()
            roots = []
            if len(factorisation) < self._poly.degree():
                print("[WARN] Not all Weierstrass points are rational.")
                K = self._poly.base_ring().extension(2, 'ζ')
            for factor, _ in factorisation:
                if factor.degree() == 1:
                    roots.append(-factor[0])
                else:
                    roots += factor.roots(ring=K, multiplicities=False)
            if len(roots) == 5:
                roots.append(None)
            self._roots = tuple(roots)
        return self._roots

    @cached_method
    def weierstrass_points(self) -> tuple[ProjectivePoint]:
        r"""
        Return the six Weierstraß points of ``self``.
        """
        return tuple(self._curve(r, 0) if r is not None
                     else self._curve(1, 0, 0) for r in self.roots())

    @cached_method
    def quadratic_splittings(self) -> tuple[tuple[Polynomial_ZZ_pEX]]:
        """
        Return the 15 quadratic splittings of ``self`` as a tuple.

        OUTPUT: A tuple of 15 quadratic splittings, where each entry is itself
        a tuple containing either 3 quadratic or 2 quadratic and 1 linear
        factors.
        """
        result = []
        for π in [(0, 1, 2, 3, 4, 5),
                  (0, 1, 2, 4, 3, 5),
                  (0, 1, 2, 5, 3, 4),
                  (0, 2, 1, 3, 4, 5),
                  (0, 2, 1, 4, 3, 5),
                  (0, 2, 1, 5, 3, 4),
                  (0, 3, 1, 2, 4, 5),
                  (0, 3, 1, 4, 2, 5),
                  (0, 3, 1, 5, 2, 4),
                  (0, 4, 1, 2, 3, 5),
                  (0, 4, 1, 3, 2, 5),
                  (0, 4, 1, 5, 2, 3),
                  (0, 5, 1, 2, 3, 4),
                  (0, 5, 1, 3, 2, 4),
                  (0, 5, 1, 4, 2, 3)]:
            # Assume that all Weierstrass points are rational
            _, _, x = self._Fx
            raw = [(x - self.roots()[i]) if self.roots()[i] is not None
                   else 1 for i in π]
            output = tuple(raw[2 * i] * raw[2 * i + 1] for i in range(3))
            result.append(output)
        return tuple(result)

    @cached_method
    def max_iso_22_subgroup_gens(self) -> tuple[
        tuple[MumfordDivisorClassField]
    ]:
        r"""
        Return a basis for each of the 15 `(2,2)`-subgroups of ``self``.

        A `(2,2)`-subgroup is a maximal isotropic subgroup of the 2-torsion
        subgroup of ``self`` with respect to the canonical polarisation.
        """
        return tuple((self(qs[0], 0), self(qs[1], 0))
                     for qs in self.quadratic_splittings())

    def transform_to_type_1(
        self,
        kernel: tuple[MumfordDivisorClassField |
                      tuple[Polynomial_ZZ_pEX]] = (),
        _4_torsion_point: (MumfordDivisorClassField |
                           tuple[Polynomial_ZZ_pEX] | None) = None
    ) -> PPAS_Isomorphism:
        r"""
        Return an isomorphism from ``self`` to the Jacobian of some genus-2
        hyperelliptic curve given by a type 1 equation, as in [Kunzweiler24].

        INPUT:

        - ``kernel`` -- (default: ``()``) a `(2,2)`-subgroup of ``self``
          specified by one of:
            - the empty tuple ``()``, which corresponds to the first entry of
              ``self.quadratic_splittings()``.
            - a quadratic splitting ``(g1, g2, g3)``, as returned by
              ``self.quadratic_splittings()``.
            - a pair of Mumford divisors ``((g1, 0), (g2, 0))``.
            - a triple of Mumford divisors ``((g1, 0), (g2, 0), (g3, 0))``.

        - ``_4_torsion_point`` -- (default: None) an optional 4-torsion Mumford
          divisor `T` satisfying `2T=\mathcal(J)_{\mathcal{C}}(g_1,0)`, used
          to avoid a square root computation.

        OUTPUT: the isomorphism in Proposition 9 of [Kunzweiler24].
        """

        if len(kernel) == 0:
            qs = self.quadratic_splittings()[0]
        elif all(isinstance(P, Polynomial_ZZ_pEX) for P in kernel):
            qs = kernel
        elif len(kernel) == 2:
            qs = (kernel[0][0], kernel[1][0], (kernel[0] + kernel[1])[0])
        elif len(kernel) == 3:
            qs = [P[0] for P in kernel]

        g1, g2, g3 = qs
        F, _, x = self._Fx

        if self._poly.degree() == 6:
            α1, α2 = sorted(g1.roots(multiplicities=False))
            Δ = α2 - α1
            δ = ~Δ
            m1 = (F(1), -α2, F(1), -α1)  # Möbius transformation coefficients
        else:
            if g2.degree() == 1:
                g1, g2 = g2, g1
            elif g3.degree() == 1:
                g1, g3 = g3, g1
            α = -g1[0] / g1[1]
            Δ = δ = F(1)
            m1 = (F(1), -α, F(0), F(1))  # Möbius transformation coefficients

        def pullback(m, p, n):
            a_, b_, c_, d_ = m
            num, den = d_ * x - b_, a_ - c_ * x
            return sum(p[k] * num ** k * den ** (n - k) for k in range(n + 1))

        h2, h3 = [pullback(m1, g, 2) for g in (g2, g3)]
        η2, η3 = 1 / h2[2], 1 / h3[2]
        β, ββ = h2[0] * η2, -h2[1] * η2  # β1β2, β1+β2
        γ, γγ = h3[0] * η3, -h3[1] * η3  # γ1γ2, γ1+γ2
        cg = self.lc() * h2[2] * h3[2] * δ ** 4

        sqrtβ = None
        if _4_torsion_point:  # Need to test
            Tu, Tv = _4_torsion_point
            u = pullback(m1, Tu, 2).monic()  # It is assumed that deg(u)=2
            if u.degree() == 2:
                v = (pullback(m1, Tv, 3) * δ ** 3) % u
                a0, a1, b0, b1 = u[0], u[1], v[0], v[1]
                num = ((a0 * b0 * b1 - a1 * b0 ** 2) * β
                       + cg * a0 ** 2 * (a0 - β) ** 2)
                den = b0 ** 2 * β + cg * a0 ** 2 * (a0 - β) * (-a1 - ββ)
                sqrtβ = num / den
        if not sqrtβ:
            sqrtβ = min(β.sqrt(all=True))
        σ = a = ~sqrtβ

        E = cg * sqrtβ ** 5
        A = ββ * a
        B = γγ * a
        C = γ * a ** 2
        codomain = HyperellipticJacobian_g2_generic(
            E * x * (x ** 2 - A * x + 1) * (x ** 2 - B * x + C),
            type_1_inv=(A, B, C, E)
        )

        m, ε = (a * m1[0], a * m1[1], m1[2], m1[3]), (δ * sqrtβ) ** 3

        def φ(P: MumfordDivisorClassField):
            Pu, Pv = P.uv()
            w = P._n

            if Pu.degree() == 0:
                if self._poly.degree() == 5 or w == 1:
                    return codomain()
                Q = codomain._curve(σ, self.sqrt_lc())
                R = codomain._curve(σ, -self.sqrt_lc())
                return codomain(Q, R) if w == 2 else codomain(R, Q)

            if Pu.degree() == 1:
                a, b = -Pu[0], Pv[0]
                χ = m[2] * a + m[3]  # denominator
                if χ == 0:  # image at infinity
                    Q = codomain._curve(1, 0, 0)
                else:  # affine image
                    δ = ~χ
                    a_, b_ = (m[0] * a + m[1]) * δ, b * δ ** 3
                    Q = codomain._curve(a_, b_)
                if self._poly.degree() == 5:
                    return codomain(Q)
                y = self.sqrt_lc() if w == 0 else -self.sqrt_lc()
                R = codomain._curve(σ, y)
                return codomain(Q, R)
            u = pullback(m, Pu, 2).monic()
            v = (pullback(m, Pv, 3) * ε) % u
            # if u.degree() == 1:
            #     # P is supported on two points, but one is mapped to ∞ in the
            #     # codomain curve
            #     a = -Pu[1] + m1[3]/m1[2]
            #     b = Pv(a)
            #     Q = codomain._curve(a,b)
            #     return codomain(Q)
            return codomain(u, v)
        return self.isomorphism(codomain, φ)

    def isomorphism(
        self,
        codomain: HyperellipticJacobian_g2_generic,
        map: Callable[[MumfordDivisorClassField], MumfordDivisorClassField]
    ) -> PPAS_Isomorphism:
        """
        Manually create an isomorphism from `self` to `codomain`.
        """
        return PPAS_Isomorphism(self, codomain, map)

    @cached_method
    def invariants(self) -> tuple[FiniteRingElement]:
        r"""
        Return the absolute Igusa invariants of ``\mathcal{C}``.
        """
        return self._curve.absolute_igusa_invariants_kohel()

    def lift_x(self,
               x0: FiniteRingElement,
               weight: Integer = 1) -> MumfordDivisorClassField:
        r"""
        Lift `x_0\in\mathbb{F}_q` to a divisor class `[(x_0,y_0)-O]`
        where `O` is a point at infinity.

        INPUT:

        - ``x0`` -- the `x`-coordinate of a point `P` on `\mathcal{C}`.

        - ``weight` -- (default: 1) if `f(x)` has degree 5, ``weight`` is
          ignored; otherwise, ``weight=0`` and ``weight=1`` specifies that
          the subtracted point at infinity is `\infty_-` and `\infty_+`
          respectively.

        OUTPUT: The Mumford divisor `[P-\infty]` if `f(x)` has degree 5, or
        `[P-\infty_\pm]` if `f(x)` has degree 6.
        """
        a, b, _ = self._curve.lift_x(x0)
        _, _, x = self._Fx
        if self._poly.degree() == 6:
            return self(x - a, b, weight)
        else:
            return self(x - a, b)

    def lift_xx(self,
                x1: FiniteRingElement,
                x2: FiniteRingElement) -> MumfordDivisorClassField:
        r"""
        Lift `x_1,x_2\in\mathbb{F}_q` to a divisor class
        `[(x_1,y_1)-(x_2,y_2)]\in\mathcal{J}_{\mathcal{C}}`.
        """
        if x1 == x2:
            return self()
        return self(self._curve.lift_x(x1), self._curve.lift_x(x2))

    @cached_method
    def sqrt_lc(self) -> FiniteRingElement:
        """
        Return the principal square root of the leading coefficient of `f(x)`,
        provided it is rational.
        """
        return min(self.lc().sqrt(all=True))

    @cached_method
    def points_at_infinity(self) -> tuple[ProjectivePoint]:
        r"""
        Return a tuple containing the one or two point(s) at infinity.

        If `f(x)` has degree 5, the single point at infinity is
        `\infty=(1:0:0)`.

        If `f(x)` has degree 6 and leading coefficient `c`, the two points at
        infinity are `\infty_\pm=(1:\pm\alpha:0)`, where `\alpha^2=c`.
        """
        if self._poly.degree() == 5:
            return (self._curve(1, 0, 0),)
        else:
            α = self.sqrt_lc()
            return (self._curve(1, α, 0), self._curve(1, -α, 0))

    def isogeny22(
        self,
        kernel_gens: tuple[MumfordDivisorClassField |
                           tuple[Polynomial_ZZ_pEX]] = (),
        _4_torsion_point: (MumfordDivisorClassField |
                           tuple[Polynomial_ZZ_pEX] | None) = None
    ) -> PPAS_Isogeny:
        r"""
        Return a `(2,2)`-isogeny from ``self`` with kernel generated by
        ``kernel_gens``.

        INPUT:

        - ``kernel_gens`` -- (default: ``()``) a `(2,2)`-subgroup of ``self``
          specified by one of:
            - an empty tuple ``()``, which corresponds to the first entry of
              ``self.quadratic_splittings()``.
            - a quadratic splitting ``(g1, g2, g3)``, as returned by
              ``self.quadratic_splittings()``.
            - a pair of Mumford divisors ``((g1, 0), (g2, 0))``.
            - a triple of Mumford divisors ``((g1, 0), (g2, 0), (g3, 0))``.

        - ``_4_torsion_point`` -- (default: None) an optional 4-torsion Mumford
          divisor `T` satisfying `2T=\mathcal{J}_{\mathcal{C}}(g_1,0)`, used
          to avoid a square root computation.
        """
        F, _, x = self._Fx  # for affine use
        X, Y, Z = polygens(F, 'X,Y,Z')  # for projective use
        inv = self.type_1_inv
        if inv is None:
            γ = self.transform_to_type_1(kernel_gens, _4_torsion_point)
            φ = γ.codomain.isogeny22()
            return γ.post_compose(φ, kernel_gens)
        A, B, C, E = inv
        if C != 1:  # Richelot isogeny
            δ = 1 / (E * (1 - C))
            Aʼ, Bʼ, Cʼ, Eʼ = C, 2 / E, (B - A * C) * δ, (A - B) * δ
            type_2_inv = Aʼ, Bʼ, Cʼ, Eʼ
            f = (x ** 2 - 1) * (x ** 2 - Aʼ) * (Eʼ * x ** 2 - Bʼ * x + Cʼ)
            codomain = HyperellipticJacobian_g2_generic(
                f, type_2_inv=type_2_inv
            )

            def φ(P: MumfordDivisorClassField | tuple[Polynomial_ZZ_pEX]) \
                    -> MumfordDivisorClassField:
                a, b = P
                a0, a1, a2, b0, b1 = a[0], a[1], a[2], b[0], b[1]
                if a2 == 0:
                    # degree ≤1
                    if a1 == 0:
                        # degree 0
                        return codomain()
                    elif b0 == 0:
                        # Weierstrass point; Proposition 12
                        if a0 == 0:
                            return codomain()
                        if a0 ** 2 + A * a0 + 1 == 0:
                            return codomain(x ** 2 - 1, 0)
                        if a0 ** 2 + B * a0 + C == 0:
                            return codomain(x ** 2 - Aʼ, 0)
                        raise ValueError(
                            "The Mumford divisors of P are invalid.")
                    elif a0 ** 2 + B * a0 + 1 == 0:
                        # Point at infinity; Proposition 21
                        v2 = (B - A) * a0 / b0
                        inv2 = ~F(2)
                        ap = x - B * inv2
                        bp = (v2 * (B ** 2 - 4 * C) * (B + 2 * a0) *
                              inv2 ** 3)
                        n = 1 if v2 < -v2 else 0
                        JP = codomain(ap, bp, n)  # This is DP – D∞
                        JQ = codomain(x ** 2 - C, 0)  # This is DQ – D∞
                        return JP + JQ
                    else:
                        # Non-Weierstrass point; Proposition 14
                        u = -a0
                        u2 = u ** 2
                        u3 = u2 * u
                        Bu = B * u
                        δ = 1 / (u2 - Bu + 1)
                        ap1 = 2 * (C - 1) * u * δ
                        ap0 = (-C * u2 + Bu - C) * δ
                        k = u * (1 - C) * (u2 - A * u + 1) * δ * δ / b0
                        bp1 = k * (
                            2 * u3 - B * u2 + (-B ** 2 + 4 * C - 2) * u + B
                        )
                        bp0 = -k * (
                            B * u3 + (-B ** 2 + 2 * C) * u2 - Bu + 2 * C
                        )
                        JP = codomain(x ** 2 + ap1 * x + ap0,
                                      bp1 * x + bp0)  # This is DP – D∞
                        JQ = codomain(x ** 2 - Aʼ, 0)  # This is DQ – D∞
                        return JP + JQ
                else:
                    # degree 2
                    if b == 0:
                        # 2 Weierstrass points; Extension of Proposition 12
                        if (a1, a0) in [
                                (-A, 1), (-B, C)]:  # Kernel element
                            return codomain()
                        if a1 ** 2 - 4 * a0 == 0:  # 2 identical points
                            return codomain()
                        resα = (a0 * A ** 2 + (a0 + 1) * a1 * A
                                + (a0 - 1) ** 2 + a1 ** 2)
                        resβ = (a0 * B ** 2 + (a0 + C) * a1 * B
                                + (a0 - C) ** 2 + C * a1 ** 2)
                        JP = (codomain(x ** 2 - 1, 0) if resα == 0
                              else codomain())
                        JQ = (codomain(x ** 2 - Aʼ, 0) if resβ == 0
                              else codomain())
                        return JP + JQ
                    # Point at infinity
                    elif (a0 * B ** 2 + (a0 + 1) * a1 * B + (a0 - 1) ** 2
                          + a1 ** 2 == 0):
                        if (a1, a0) == (-B, 1):
                            # Proposition 24
                            if B == 2:
                                v = (A - 2) / b0
                                n = 0 if v < -v else 1
                                return 2 * codomain(x - 1, 0, n)
                            elif B == -2:
                                v = (A - 2) / b0
                                n = 1 if v < -v else 0
                                return 2 * codomain(x + 1, 0, n)
                            elif b0 == 0:
                                v = (A - B) * b1
                                return codomain(1, 0, 2 if v < -v else 0)
                            else:
                                inv2 = ~F(2)
                                ap = (x - B * inv2) ** 2
                                bp = (4 * C - B ** 2) * (B - A) * inv2 ** 2
                                return codomain(ap, bp / b0)
                        else:
                            # Proposition 23
                            s = -gcd(a, x ** 2 - B * x + 1)[0]
                            t = b1 * s + b0
                            inv2 = ~F(2)
                            ap = x - B * inv2
                            v2 = (A - B) * s / t
                            bp = ((B ** 2 - 4 * C) * (B - 2 * s) * v2
                                  * inv2 ** 3)
                            n = 1 if v2 < -v2 else 0
                            JQ = codomain(ap, bp, n)

                            u = a0 / s
                            if u == s:
                                JP = JQ
                            elif self._poly(u) == 0:
                                r = -b0 / b1
                                ap = (x ** 2 - Aʼ
                                      if r == 0
                                      else Eʼ * x ** 2 - Bʼ * x + Cʼ
                                      if r ** 2 - A * r + 1 == 0
                                      else 1)
                                JP = codomain(ap, 0)
                            else:
                                u2 = u ** 2
                                u3 = u2 * u
                                Bu = B * u
                                δ = 1 / (u2 - Bu + 1)
                                ap1 = 2 * (C - 1) * u * δ
                                ap0 = (-C * u2 + Bu - C) * δ
                                k = u * (1 - C) * (u2 - A *
                                                   u + 1) * δ * δ / b(u)
                                bp1 = k * (2 * u3 - B * u2 +
                                           (-B ** 2 + 4 * C - 2) * u + B)
                                bp0 = (B * u3 + (-B ** 2 + 2 * C) * u2
                                       - Bu + 2 * C) * -k
                                JP = codomain(x ** 2 + ap1 * x + ap0,
                                              bp1 * x + bp0)  # DP – D∞
                            return JP + JQ
                    elif -b1 * (a1 * b0 - a0 * b1) + b0 ** 2 == 0:
                        # 1 Weierstrass and 1 non-Weierstrass point;
                        # Corollary 19
                        r = -b0 / b1
                        u = -a1 - r  # modified from the paper
                        v = b(u)  # modified from the paper
                        u2 = u ** 2
                        u3 = u2 * u
                        Bu = B * u
                        δ = 1 / (u2 - Bu + 1)
                        ap1 = 2 * (C - 1) * u * δ
                        ap0 = (-C * u2 + Bu - C) * δ
                        k = u * (1 - C) * (u2 - A * u + 1) * δ * δ / v
                        bp1 = k * (2 * u3 - B * u2 +
                                   (-B ** 2 + 4 * C - 2) * u + B)
                        bp0 = -k * (B * u3 + (-B ** 2 + 2 * C)
                                    * u2 - Bu + 2 * C)
                        aq = (x ** 2 - Aʼ
                              if r == 0
                              else Eʼ * x ** 2 - Bʼ * x + Cʼ
                              if r ** 2 - A * r + 1 == 0
                              else 1)
                        JP = codomain(x ** 2 + ap1 * x + ap0,
                                      bp1 * x + bp0)  # DP – D∞
                        JQ = codomain(aq, 0)
                        return JP + JQ
                    elif 4 * a0 == a1 ** 2:
                        # Shared support; Proposition 26
                        inv2 = ~F(2)
                        u = -a1 * inv2
                        u2 = u ** 2
                        u3 = u2 * u
                        Bu = B * u
                        δ = 1 / (u2 - Bu + 1)
                        ap1 = 2 * (C - 1) * u * δ
                        ap0 = (-C * u2 + Bu - C) * δ
                        k = u * (1 - C) * (u2 - A * u + 1) * δ * δ / b0
                        bp1 = k * (
                            2 * u3 - B * u2 + (-B ** 2 + 4 * C - 2) * u + B
                        )
                        bp0 = -k * (
                            B * u3 + (-B ** 2 + 2 * C) * u2 - Bu + 2 * C
                        )
                        return codomain(x ** 2 + ap1 * x + ap0,
                                        bp1 * x + bp0) * 2  # [2P – 2D∞]
                    elif a0 == 1:
                        # Shared support; Proposition 27
                        assert a1 not in [2, -A, -B]
                        α = b1 - a1 * b0
                        β = a1 + B
                        γ = 2 * b0 * (C - 1)
                        d0 = (B * α + 2 * b0 * C) * β - B * γ
                        δ1 = 1 / ((2 * α + b0 * B) * β - 2 * γ)
                        y = (B ** 2 - 4 * C) * (C - 1) * (a1 + A) * δ1
                        P = codomain._curve(d0 * δ1, y)
                        return codomain(P, 2)  # [2P – D∞]
                    else:
                        # 2 non-Weierstrass points; Theorem 15
                        μ = a1 * b0 - a0 * b1
                        t1 = (a0 - 1) ** 2 + a1 ** 2
                        t2 = (a0 + 1) * a1
                        # a0ʼ = C * (C * t1 + B * t2) + B ** 2 * a0
                        a1ʼ = (C - 1) * (2 * C * t2 + 4 * B * a0)
                        a2ʼ = (-2 * C * t1 - B * (C + 1) * t2
                               + 2 * (2 * (C - 1) ** 2 - B ** 2) * a0)
                        a3ʼ = 2 * (1 - C) * (t2 + 2 * B * a0)
                        a4ʼ = t1 + B * (t2 + B * a0)
                        b0ʼ = (μ * (C * (t1 + A * a1) + B * a0 * (a1 + A))
                               + a0 * b0 * (a0 - 1) * (A * C - B))
                        b1ʼ = (a0 * (2 * (C - 1) * (a1 * μ - a0 * b0)
                                     + ((C - 2) * A + B) * μ
                                     + (A * B - 2) * b0 + B * b1)
                               + C * (A * ((t2 - a1) * b0 + μ)
                                      + b0 * (t1 + 2 * a0)))
                        b2ʼ = (-μ * (t1 + B * (t2 - a1)
                                     + A * (B * a0 + a1))
                               + a0 * ((B - A) * (a0 - 1) * b0
                                       + 2 * (C - 1) * (b0 * A + b1 + μ)))
                        b3ʼ = (((A * a0 + t2) * B + t1) * -b0
                               + (B - A) * (μ + a0 ** 2 * b1))
                        bden = (1 - a0) * (b0 ** 2 - b1 * μ)
                        aʼ = (a4ʼ * x ** 4 + a3ʼ * x ** 3 +
                              a2ʼ * x ** 2 + a1ʼ * x + a0) / a4ʼ
                        bʼ = (b3ʼ * x ** 3 + b2ʼ * x ** 2
                              + b1ʼ * x + b0ʼ) / bden
                        V = codomain.point_homset()
                        u, v, _ = V.cantor_reduction(aʼ, bʼ, 0)
                        return codomain(u, v, 0)

            kernel_gens = (self(x, 0), self(x ** 2 - A * x + 1, 0))
            return PPAS_22Isogeny(self, codomain, kernel_gens, φ)

        else:  # Split isogeny
            ψ1 = EllipticCurve_from_cubic(
                Y ** 2 * Z - E * (X + 2 * Z) * (X - A * Z) * (X - B * Z)
            )
            ψ2 = EllipticCurve_from_cubic(
                Y ** 2 * Z - E * (X - 2 * Z) * (X - A * Z) * (X - B * Z)
            )
            E1, E2 = ψ1.domain(), ψ2.domain()
            E1ʼ: EllipticCurve_finite_field = ψ1.codomain()
            E2ʼ: EllipticCurve_finite_field = ψ2.codomain()
            codomain = EllipticProduct(E1ʼ, E2ʼ)

            def π1(P):
                if P[0] != 0 and P[2] != 0:
                    x = P[0] + 1 / P[0]
                    y = P[1] * (P[0] + 1) / P[0] ** 2
                    return E1(x, y, 1)
                else:
                    return E1(0)

            def π2(P):
                if P[0] != 0 and P[2] != 0:
                    x = P[0] + 1 / P[0]
                    y = P[1] * (P[0] - 1) / P[0] ** 2
                    return E2(x, y, 1)
                else:
                    return E2(0)

            def φ(P: MumfordDivisorClassField) -> MumfordDivisorClassField:
                # TODO: Optimise
                result = []
                supp = self.positive_support(P)
                if not supp:
                    return codomain()
                for Q, n in supp.items():
                    for _ in range(n):
                        result.append(codomain(ψ1(π1(Q)), ψ2(π2(Q))))
                return sum(result)

            kernel_gens = (self(x, 0), self(x ** 2 - A * x + 1, 0))
            return PPAS_22Isogeny(self, codomain, kernel_gens, φ)

    @cached_method
    def support(self, P: MumfordDivisorClassField):
        u, v = P.uv()
        d = u.degree()
        roots = u.roots()
        if sum(m for _, m in roots) < u.degree():
            print("[WARN] Field extension required.")
            K = u.base_ring().extension(2, 'ζ')
            roots = u.roots(ring=K)
            C = self._curve.change_ring(K)
        else:
            C = self._curve
        support = {C(r, v(r)): m for r, m in roots}
        if self._poly.degree() == 6:
            O1, O2 = self.points_at_infinity()
            weight = P._n
            if d == 2:
                support[O1] = support[O2] = -1
            elif d == 1:
                if weight == 1:
                    support[O2] = -1
                else:
                    support[O1] = -1
            else:
                if weight == 2:
                    support[O1] = 1
                    support[O2] = -1
                elif weight == 1:
                    support = dict()
                elif weight == 0:
                    support[O1] = -1
                    support[O2] = 1
        else:
            (O,) = self.points_at_infinity()
            support[O] = -d
        return support

    @cached_method
    def positive_support(self, P: MumfordDivisorClassField):
        u, v = P.uv()
        d = u.degree()
        roots = u.roots()
        if sum(m for _, m in roots) < u.degree():
            print("[WARN] Field extension required.")
            K = u.base_ring().extension(2, 'ζ')
            roots = u.roots(ring=K)
            C = self._curve.change_ring(K)
        else:
            C = self._curve
        positive_support = {C(r, v(r)): m for r, m in roots}
        if self._poly.degree() == 6:
            O1, O2 = self.points_at_infinity()
            weight = P._n
            if d == 1:
                if weight == 1:
                    positive_support[O1] = 1
                else:
                    positive_support[O2] = 1
            elif d == 0:
                if weight == 2:
                    positive_support[O1] = 2
                elif weight == 1:
                    positive_support = dict()
                elif weight == 0:
                    positive_support[O2] = 2
        return positive_support


class EllipticProduct(CartesianProduct):
    r"""
    The Cartesian product `\mathcal{E}_1\times\mathcal{E}_2`, where
    each factor is an elliptic curve over `\mathbb{F}_q`.

    INPUT:

    - ``E1`` -- an elliptic curve `\mathcal{E}_1/\mathbb{F}_q`.

    - ``E2`` -- an elliptic curve `\mathcal{E}_2/\mathbb{F}_q`.

    - ``custom_name`` -- (default: None) a custom string representation for
      this object.

    - ``roots`` -- (default: None) a nested tuple of the form
      ``(roots1, roots2)``, where ``roots1`` and ``roots2`` are tuples
      containing the roots of `\mathcal{E}_1` and `\mathcal{E}_2` over
      `\overline{\mathbb{F}}_q` respectively. If specified, computations
      involving the roots will use this datum, avoiding polynomial
      factorisations.
    """

    def __init__(self,
                 E1: EllipticCurve_finite_field | None = None,
                 E2: EllipticCurve_finite_field | None = None,
                 **kwargs):
        r"""
        Create the Cartesian product `\mathcal{E}_1\times\mathcal{E}_2`, where
        each factor is an elliptic curve over `\mathbb{F}_q`.

        INPUT:

        - ``E1`` -- an elliptic curve `\mathcal{E}_1/\mathbb{F}_q`.

        - ``E2`` -- an elliptic curve `\mathcal{E}_2/\mathbb{F}_q`.

        - ``custom_name`` -- (default: None) a custom string representation for
          this object.

        - ``roots`` -- (default: None) a nested tuple of the form
          ``(roots1, roots2)``, where ``roots1`` and ``roots2`` are tuples
          containing the roots of `\mathcal{E}_1` and `\mathcal{E}_2` over
          `\overline{\mathbb{F}}_q` respectively. If specified, computations
          involving the roots will use this datum, avoiding polynomial
          factorisations.
        """

        if (F := E1.base_ring()) != E2.base_ring():
            raise ValueError(
                "Elliptic curves must be defined over the same field."
            )
        super().__init__((E1, E2),
                         CommutativeAdditiveGroups().CartesianProducts())
        self.E1 = E1
        self.E2 = E2
        self._roots = kwargs.get('roots')
        if custom_name := kwargs.get('custom_name'):
            self.rename(custom_name)

        self._F = F
        Fx = PolynomialRing(F, 'x')
        self._Fx = F, Fx, Fx.gen()

    @cached_method
    def zero(self) -> MumfordDivisorClassField:
        r"""
        Return the identity of the underlying abelian group.

        Alias of ``self()``.
        """
        return self()

    @cached_method
    def base_ring(self) -> FiniteField:
        r"""
        Return the finite field `\mathbb{F}_q` which ``self`` is defined over.
        """
        return self._F

    base_field = base_ring

    def change_ring(self, K: FiniteField, **kwargs) -> EllipticProduct:
        """
        Return the base change of ``self`` to the field `K`.

        INPUT:

        - ``**kwargs`` -- additional keyword arguments to pass to the
          ``HyperellipticJacobian_g2_generic`` constructor.
        """
        return EllipticProduct(self.E1.change_ring(K),
                               self.E2.change_ring(K),
                               **kwargs)

    base_extend = change_ring

    def __call__(self, *args: tuple) -> tuple[CartesianProduct.element_class]:
        r"""
        Return a point of ``self``, specified by ``*args``.

        INPUT:

        - ``args`` can take one of several forms:
            - ``()`` -- an empty tuple.
            - ``(P, Q)`` -- a pair of points on ``self.E1`` and ``self.E2``
              respectively.

          Each point `(X:Y:Z)\in\mathcal{E}_i` can be specified as:
            - an ``EllipticCurvePoint_finite_field`` object.
            - ``0``, if `Z=0`.
            - its affine coordinates ``(X, Y)``, if `Z=1`.
            - its projective coordinates ``(X, Y, Z)``.

        OUTPUT:

        - If ``args`` is ``()``, return the group identity of
          `\mathcal{E}_1\times\mathcal{E}_2`.
        - If ``args`` is ``(P, Q)``, return the Cartesian product of `P` and
          `Q`.

        EXAMPLES::

            sage: F.<ω> = GF((6143, 2), 'ω', x^2 + 1)
            sage: E1 = EllipticCurve(F, [1, 0]); E1
            Elliptic Curve defined by y^2 = x^3 + x over Finite Field in ω of
            size 6143^2
            sage: E2 = EllipticCurve(F, [0, -1]); E2
            Elliptic Curve defined by y^2 = x^3 + 6142 over Finite Field in ω
            of size 6143^2
            sage: A2 = EllipticProduct(E1, E2); A2
            The Cartesian product of (Elliptic Curve defined by y^2 = x^3 + x
            over Finite Field in ω of size 6143^2, Elliptic Curve defined by
            y^2 = x^3 + 6142 over Finite Field in ω of size 6143^2)
            sage: A2((1, 2644), (ω, 5386*ω + 353))
            ((1 : 2644 : 1), (ω : 5386*ω + 353 : 1))
            sage: A2()
            ((0 : 1 : 0), (0 : 1 : 0))
        """
        if not args:
            return super().__call__((0, 0))
        return super().__call__(args)

    @cached_method
    def torsion_gens(self, n: Integer) -> (
        tuple[CartesianProduct.element_class]
    ):
        r"""
        Return the four generators of the `n`-torsion subgroup of ``self``.

        Raise an :exc:`ValueError` if ``n`` is not a positive integer or if
        ``self`` is not an elliptic product.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if n == 1:
            return (self(),)

        O1, O2 = self()
        if n == 2:
            P1, Q1, _ = self.E1_points_of_order_2()
            P2, Q2, _ = self.E2_points_of_order_2()
        else:
            P1, Q1 = self.E1.torsion_basis(n)
            P2, Q2 = self.E2.torsion_basis(n)

        return tuple(
            self(x, y) for (x, y) in [(P1, O2), (Q1, O2), (O1, P2), (O1, Q2)]
        )

    @cached_method
    def points_of_order_2(self) -> tuple[CartesianProduct.element_class]:
        """
        Return the 15 points of order 2 of ``self``.
        """
        W = self.weierstrass_points()
        return tuple(self(W[0][i], W[1][j])
                     for i in range(4) for j in range(4) if i + j != 0)

    @cached_method
    def E1_points_of_order_2(self) -> tuple[EllipticCurvePoint_finite_field]:
        r"""
        Return the three of order 2 of `\mathcal{E}_1`.
        """
        return tuple(self.E1(r, 0, 1) for r in self.roots[0])

    @cached_method
    def E2_points_of_order_2(self) -> tuple[EllipticCurvePoint_finite_field]:
        r"""
        Return the three points of order 2 of `\mathcal{E}_2`.
        """
        return tuple(self.E2(r, 0, 1) for r in self.roots[1])

    @cached_method
    def roots(self) -> tuple[FiniteRingElement]:
        r"""
        Return the roots of ``self``.

        OUTPUT:

        - A nested tuple ``(roots1, roots2)``, where ``roots1`` and ``roots2``
          are tuples containing the 3 roots of `\mathcal{E}_1` and
          `\mathcal{E}_2` respectively.
        """
        if self._roots is None:
            F = [E.hyperelliptic_polynomials()[0]
                 for E in self.cartesian_factors()]
            self._roots = tuple(tuple(f.roots(multiplicities=False))
                                for f in F)
        return self._roots

    @cached_method
    def weierstrass_points(self) -> (
        tuple[tuple[EllipticCurvePoint_finite_field]]
    ):
        r"""
        Return the four Weierstraß points of ``\mathcal{E}_1`` and
        ``\mathcal{E}_2`` respectively, as a nested tuple.
        """
        return ((self.E1(0),) + self.E1_points_of_order_2,
                (self.E2(0),) + self.E2_points_of_order_2)

    @cached_method
    def _22_subgroup_gens(self) -> tuple[
        tuple[CartesianProduct.element_class]
    ]:
        r"""
        Return a basis for each of the 35 2-torsion subgroups of ``self``
        isomorphic to `(\mathbb{Z}/2\mathbb{Z})^2`.
        """
        result = []
        T = self.torsion_gens(2)
        for V in VectorSpace(GF(2), 4).subspaces(2):
            v = [[Integer(g) for g in G] for G in V.gens()]
            β = tuple(dot(T, v[i]) for i in range(2))
            result.append(β)
        return tuple(result)

    @cached_method
    def max_iso_22_subgroup_gens(self) -> tuple[
        tuple[CartesianProduct.element_class]
    ]:
        r"""
        Return a basis for each of the 15 `(2,2)`-subgroups of ``self``.

        A `(2,2)`-subgroup is a maximal isotropic subgroup of the 2-torsion
        subgroup of ``self`` with respect to the product polarisation.
        """

        O1, O2 = self()
        K1, K2 = self.E1_points_of_order_2(), self.E2_points_of_order_2()
        L1, L2 = [self(R, O2) for R in K1], [self(O1, R) for R in K2]
        diagonal = tuple((R1, R2) for R1 in L1 for R2 in L2)
        permutation = tuple((self(K1[1], K2[π(1)]), self(K1[2], K2[π(2)]))
                            for π in SymmetricGroup([0, 1, 2]))
        return diagonal + permutation

    def isomorphism(
                self,
                codomain: EllipticProduct,
                map: Callable[[CartesianProduct.element_class],
                              CartesianProduct.element_class]
            ) -> PPAS_Isomorphism:
        """
        Create an isomorphism from `self` to `codomain`.
        """
        return PPAS_Isomorphism(self, codomain, map)

    @cached_method
    def invariants(self) -> tuple[FiniteRingElement] | set[FiniteRingElement]:
        r"""
        Return the unordered pair of `j`-invariants
        `\{j(\mathcal{E}_1),j(\mathcal{E}_2)\}`.
        """
        return {E.j_invariant() for E in self.cartesian_factors()}

    def lift_xx(self,
                x1: FiniteRingElement,
                x2: FiniteRingElement) -> CartesianProduct.element_class:
        r"""
        Lift `x_1,x_2\in\mathbb{F}_q` to a point `((x_1,y_1),(x_2,y_2))`.
        """
        return self(self.E1.lift_x(x1), self.E2.lift_x(x2))

    def isogeny22(
        self,
        kernel_gens: tuple[CartesianProduct.element_class]
    ) -> PPAS_Isogeny:
        r"""
        Return a `(2,2)`-isogeny from ``self`` with kernel generated by
        ``kernel_gens``.

        INPUT:

        - ``kernel_gens`` -- a pair of points generating a `(2,2)`-subgroup.
        """
        _, Fx, x = self._Fx  # for affine use
        P = None
        O1, O2 = self()
        if kernel_gens[0][0] == O1:
            (_, Q), (P, _) = kernel_gens
        elif kernel_gens[0][1] == O2:
            (P, _), (_, Q) = kernel_gens
        else:
            (P1, Q1), (P2, Q2) = kernel_gens
            P3, Q3 = sum(kernel_gens)
            α1, α2, α3 = P1[0], P2[0], P3[0]
            β1, β2, β3 = Q1[0], Q2[0], Q3[0]

        if P:  # Product isogeny
            E1 = self.E1.isogeny_codomain([P])
            E2 = self.E2.isogeny_codomain([Q])
            codomain = EllipticProduct(E1, E2)
            φ1 = self.E1.isogeny(P, E1, 2)
            φ2 = self.E2.isogeny(Q, E2, 2)

            def φ(P: CartesianProduct.element_class) \
                    -> CartesianProduct.element_class:
                return codomain(φ1(P[0]), φ2(P[1]))
            return PPAS_22Isogeny(self, codomain, kernel_gens, φ)

        # Anti-isometry
        if len(self.invariants) == 1:
            for γ in self.E1.isomorphisms(self.E2):
                if γ(P1) == Q1 and γ(P2) == Q2:
                    def φ(P: CartesianProduct.element_class) \
                            -> CartesianProduct.element_class:
                        Q, R = P
                        return self(Q + γ.inverse_image(R), γ(Q) - R)
                    return PPAS_22Isogeny(self, self, kernel_gens, φ)

        # Gluing isogeny
        M = Matrix(self.base_field, [
            [α1 * β1, α1, β1], [α2 * β2, α2, β2], [α3 * β3, α3, β3]
        ])
        M_inv = M.inverse()
        R, S, T = M_inv * vector(self.base_field, [1, 1, 1])
        R_inv = ~R
        RD_inv = R_inv * M_inv.det()
        dα = (α1 - α2) * (α2 - α3) * (α3 - α1)
        dβ = (β1 - β2) * (β2 - β3) * (β3 - β1)
        s1, t1 = -dα * RD_inv, dβ * RD_inv
        s2, t2 = -T * R_inv, -S * R_inv
        s1_inv, t1_inv = ~s1, ~t1
        α1ʼ = (α1 - s2) * s1_inv
        α2ʼ = (α2 - s2) * s1_inv
        α3ʼ = (α3 - s2) * s1_inv
        f = s1 * (x ** 2 - α1ʼ) * (x ** 2 - α2ʼ) * (x ** 2 - α3ʼ)
        codomain = HyperellipticJacobian_g2_generic(f)

        def φ(P: CartesianProduct.element_class) -> MumfordDivisorClassField:
            Q, R = P
            if (Q, R) in [(O1, O2), (P1, Q1), (P2, Q2), (P3, Q3)]:
                # kernel
                return codomain()

            if Q == O1:
                JQ = codomain()
            else:
                JQ = codomain(s1 * x ** 2 + s2 - Q[0], Fx(Q[1] * s1_inv))

            if R == O2:
                JR = codomain()
            else:
                u = (R[0] - t2) * x ** 2 - t1
                v = (R[1] * t1_inv * x ** 3) % u
                JR = codomain(u, v)

            return JQ + JR
        return PPAS_22Isogeny(self, codomain, kernel_gens, φ)


class PPAS_Isogeny(SageObject):
    r"""
    A polarised isogeny `\varphi` between principally polarised abelian
    surfaces.

    It is recommended not to manually instantiate instances of this class,
    but rather using the ``isogeny22()`` methods of ``EllipticProduct`` and
    ``HyperellipticJacobian_g2_generic``.

    Compose isogenies using the ``pre_compose`` and ``post_compose`` methods,
    and map a point `P` using the syntax ``φ(P)``.
    """    
    def __init__(
        self,
        domain: HyperellipticJacobian_g2_generic | EllipticProduct,
        codomain: HyperellipticJacobian_g2_generic | EllipticProduct,
        degrd: Integer = 1,
        kernel_gens: tuple[MumfordDivisorClassField |
                           CartesianProduct.element_class] = (),
        chain: list[PPAS_Isogeny] = [],
        map: Callable[
            [MumfordDivisorClassField | CartesianProduct.element_class],
            MumfordDivisorClassField | CartesianProduct.element_class
        ] = None
    ):
        r"""
        Create a polarised isogeny `\varphi` between principally polarised
        abelian surfaces.

        It is recommended not to manually instantiate instances of this class,
        but rather using the ``isogeny22()`` methods of ``EllipticProduct``
        and ``HyperellipticJacobian_g2_generic``.

        INPUT:

        - ``domain`` -- the ``EllipticProduct`` or
          ``HyperellipticJacobian_g2_generic`` where `\varphi` originates.

        - ``codomain`` -- the ``EllipticProduct`` or
          ``HyperellipticJacobian_g2_generic`` where `\varphi` terminates.

        - ``degrd`` -- (default: 1) the reduced degree
          `\operatorname{degrd}\varphi=\sqrt{\deg\varphi}`.

        - ``kernel_gens`` -- (default: ()) a basis for the kernel.

        - ``chain`` -- (default: []) the factorisation of `\varphi` into
          smaller isogenies (of prime reduced degree) or isomorphisms.

        - ``map`` -- (default: None) the explicit map on points of ``domain``.
        """

        self.domain = domain
        self.codomain = codomain
        self.degrd = degrd
        self.degree = degrd ** 2
        self.kernel_gens = kernel_gens
        self.chain = chain
        self.map = map

    def __repr__(self):
        if self.degrd == 1:
            return (f"Polarised isomorphism from {self.domain} "
                    f"to {self.codomain}")
        elif self.degrd == 2:
            return f"(2,2)-isogeny from {self.domain} to {self.codomain}"
        elif len(self.kernel_gens) == 2:
            return (f"({self.degrd},{self.degrd})-isogeny from {self.domain} "
                    f"to {self.codomain}")
        else:
            return (f"Isogeny of reduced degree {self.degrd} from "
                    f"{self.domain} to {self.codomain}")

    def post_compose(
        self,
        other: PPAS_Isogeny,
        kernel_gens: tuple[MumfordDivisorClassField |
                            CartesianProduct.element_class]
    ):
        """
        Post-compose the isogeny ``self`` with another isogeny ``other``.

        INPUT:

        - ``other`` -- another isogeny.

        - ``kernel_gens`` -- the kernel of the composite isogeny.

        OUTPUT: the composite isogeny which maps a point through ``self``,
        followed by ``other``.
        """
        assert self.codomain == other.domain
        return PPAS_Isogeny(self.domain, other.codomain,
                            self.degrd * other.degrd, kernel_gens,
                            self.chain + other.chain)

    def pre_compose(
        self,
        other: PPAS_Isogeny,
        kernel_gens: tuple[MumfordDivisorClassField |
                            CartesianProduct.element_class]
    ):
        """
        Pre-compose the isogeny ``self`` with another isogeny ``other``.

        INPUT:

        - ``other`` -- another isogeny.

        - ``kernel_gens`` -- the kernel of the composite isogeny.

        OUTPUT: the composite isogeny which maps a point through ``other``,
        followed by ``self``.
        """
        assert self.domain == other.codomain
        return PPAS_Isogeny(other.domain, self.codomain,
                            self.degrd * other.degrd, kernel_gens,
                            other.chain + self.chain)

    def __call__(self, P):
        """
        Return the image of the point ``P`` under the isogeny.
        """        
        for φ in self.chain:
            if hasattr(φ, 'map'):
                P = φ.map(P)
        return P

    def __getitem__(self, index):
        """
        Return the isogeny in ``self.chain`` with the specified index.
        """        
        return self.chain[index]


class PPAS_22Isogeny(PPAS_Isogeny):
    r"""
    A polarised `(2,2)`-isogeny `\varphi` between principally
    polarised abelian surfaces.

    It is recommended not to manually instantiate instances of this class,
    but rather using the ``isogeny22()`` methods of ``EllipticProduct``
    and ``HyperellipticJacobian_g2_generic``.

    INPUT:

    - ``domain`` -- the ``EllipticProduct`` or
      ``HyperellipticJacobian_g2_generic`` where `\varphi` originates.

    - ``codomain`` -- the ``EllipticProduct`` or
      ``HyperellipticJacobian_g2_generic`` where `\varphi` terminates.

    - ``kernel_gens`` -- (default: ()) a basis for the kernel.

    - ``map`` -- (default: None) the explicit map on points of ``domain``.
    """

    def __init__(
        self,
        domain: HyperellipticJacobian_g2_generic | EllipticProduct,
        codomain: HyperellipticJacobian_g2_generic | EllipticProduct,
        kernel_gens: tuple[MumfordDivisorClassField |
                           CartesianProduct.element_class],
        map: Callable[
            [MumfordDivisorClassField | CartesianProduct.element_class],
            MumfordDivisorClassField | CartesianProduct.element_class
        ]
    ):
        r"""
        Create a polarised `(2,2)`-isogeny `\varphi` between principally
        polarised abelian surfaces.

        It is recommended not to manually instantiate instances of this class,
        but rather using the ``isogeny22()`` methods of ``EllipticProduct``
        and ``HyperellipticJacobian_g2_generic``.

        INPUT:

        - ``domain`` -- the ``EllipticProduct`` or
          ``HyperellipticJacobian_g2_generic`` where `\varphi` originates.

        - ``codomain`` -- the ``EllipticProduct`` or
          ``HyperellipticJacobian_g2_generic`` where `\varphi` terminates.

        - ``kernel_gens`` -- (default: ()) a basis for the kernel.

        - ``map`` -- (default: None) the explicit map on points of ``domain``.
        """
        super().__init__(domain, codomain, 2, kernel_gens, [self], map)


class PPAS_Isomorphism(PPAS_Isogeny):
    r"""
    A polarised isomorphism `\varphi` between principally
    polarised abelian surfaces.

    INPUT:

    - ``domain`` -- the ``EllipticProduct`` or
      ``HyperellipticJacobian_g2_generic`` where `\varphi` originates.

    - ``codomain`` -- the ``EllipticProduct`` or
      ``HyperellipticJacobian_g2_generic`` where `\varphi` terminates.

    - ``map`` -- (default: None) the explicit map on points of ``domain``.
    """

    def __init__(
        self,
        domain: HyperellipticJacobian_g2_generic | EllipticProduct,
        codomain: HyperellipticJacobian_g2_generic | EllipticProduct,
        map: Callable[
            [MumfordDivisorClassField | CartesianProduct.element_class],
            MumfordDivisorClassField | CartesianProduct.element_class
        ]
    ):
        r"""
        Create a polarised isomorphism `\varphi` between principally
        polarised abelian surfaces.

        INPUT:

        - ``domain`` -- the ``EllipticProduct`` or
          ``HyperellipticJacobian_g2_generic`` where `\varphi` originates.

        - ``codomain`` -- the ``EllipticProduct`` or
          ``HyperellipticJacobian_g2_generic`` where `\varphi` terminates.

        - ``map`` -- (default: None) the explicit map on points of ``domain``.
        """
        super().__init__(domain, codomain, 1, (), [self], map)


class IKO(SageObject):
    r"""
    A helper class to prepare for computations involving the Ibukiyama--Katsura
    --Oort correspondence.

    INPUT:

    - ``p`` -- a prime `p` such that `p\equiv 7` (mod 8).

    OUTPUT: a tuple containing the following useful data for working with the
    IKO correspondence:

    - ``iko`` -- this ``IKO`` object.

    - ``p`` -- the provided prime `p`.

    - ``F, ω`` -- the finite field `F=\mathbb{F}_{p^2}=\mathbb{F}_p[\omega]/
      \langle\omega^2+1\rangle` and its generator.

    - ``Fx, x`` -- the polynomial ring `F[x]=\mathbb{F}_{p^2}[x]` and its
      generator.

    - ``Hp`` -- the quaternion algebra `\mathbb{H}_p=(-1,-p \mid \mathbb{Q})`.

    - ``O0`` -- the maximal quaternion order
      `\operatorname{End}\mathcal{E}_0\cong\mathcal{O}_0\subset\mathbb{H}_p`.

    - ``E0`` -- the supersingular elliptic curve `\mathcal{E}_0/F:y^2=x^3+x`.

    - ``A0`` -- the superspecial abelian surface
      `\mathcal{A}_0=\mathcal{E}_0\times\mathcal{E}_0`.

    - ``ι`` -- the automorphism `(x,y)\mapsto(-x,iy)` on `\mathcal{E}_0`,
      corresponding to `i\in\mathbb{H}_p`.

    - ``π`` -- the Frobenius automorphism `(x,y)\mapsto(x^p,y^p)` on
      `\mathcal{E}_0`, corresponding to `j\in\mathbb{H}_p`.

    - ``_22_matrices`` -- a list of 35 matrices in
      `\mathbb{M}_2(\mathcal{O}_0)`, encoding `(2,2)`-isogenies from
      `\mathcal{E}_0`.

    - ``I2`` -- the `2\times 2` identity matrix in
      `\mathbb{M}_2(\mathcal{O}_0)`.

    - ``matrix_space`` -- the `MatrixSpace` `\mathbb{M}_2(\mathbb{H}_p)`.

    - ``IKO_Matrix`` -- a factory function to instantiate a matrix in
      `\mathbb{M}_2(\mathbb{H}_p)` from its entries.

    - ``i, ζ, ξ`` -- generators for `\mathcal{O}_0=\mathbb{Z}[i,\zeta,\xi]`.
      Explicitly, `\zeta=\frac{i+j}{2}` and `\xi=\frac{1+k}{2}`.

    EXAMPLES::

        sage: (iko, p, F, ω, Fx, x, Hp, O0, E0, A0, ι, π, _22_matrices, I2,
        ....:  matrix_space, IKO_Matrix, i, ζ, ξ) = IKO(6143).dump()
    """
    def __init__(self, p: Integer):
        x = polygen(ZZ)
        self.p = p
        F = GF((p, 2), 'ω', x ** 2 + 1)
        Fx = PolynomialRing(F, 'x')
        Hp = QuaternionAlgebra(p)
        ω, x, (i, j, k) = F.gen(), Fx.gen(), Hp.gens()
        ζ, ξ = (i + j) / 2, (1 + k) / 2
        self.F = F
        self.Fx, self.x, self.ω = Fx, x, ω
        self.Hp = self.quaternion_algebra = Hp
        self.O0 = self.quaternion_order = Hp.maximal_order(
            order_basis=(Hp(1), i, ζ, ξ))
        self.E0 = E0 = EllipticCurve(F, [0, 0, 0, 1, 0])
        self.E0.rename("E₀")
        self.A0 = EllipticProduct(E0, E0,
                                  roots=((0, -ω, ω), (0, -ω, ω)),
                                  custom_name="A₀")
        self.matrix_space = MatrixSpace(Hp, 2)

    def dump(self) -> tuple:
        r"""
        Return the following useful data for working with the IKO
        correspondence:

        - ``iko`` -- this ``IKO`` object.

        - ``p`` -- the provided prime `p`.

        - ``F, ω`` -- the finite field `F=\mathbb{F}_{p^2}=
          \mathbb{F}_p[\omega]/\langle\omega^2+1\rangle` and its generator.

        - ``Fx, x`` -- the polynomial ring `F[x]=\mathbb{F}_{p^2}[x]` and
          its generator.

        - ``Hp`` -- the quaternion algebra
          `\mathbb{H}_p=(-1,-p \mid \mathbb{Q})`.

        - ``O0`` -- the maximal quaternion order `\operatorname{End}
          \mathcal{E}_0\cong\mathcal{O}_0\subset\mathbb{H}_p`.

        - ``E0`` -- the supersingular elliptic curve
          `\mathcal{E}_0/F:y^2=x^3+x`.

        - ``A0`` -- the superspecial abelian surface
          `\mathcal{A}_0=\mathcal{E}_0\times\mathcal{E}_0`.

        - ``ι`` -- the automorphism `(x,y)\mapsto(-x,iy)` on `\mathcal{E}_0`,
          corresponding to `i\in\mathbb{H}_p`.

        - ``π`` -- the Frobenius automorphism `(x,y)\mapsto(x^p,y^p)` on
          `\mathcal{E}_0`, corresponding to `j\in\mathbb{H}_p`.

        - ``_22_matrices`` -- a list of 35 matrices in
          `\mathbb{M}_2(\mathcal{O}_0)`, encoding `(2,2)`-isogenies from
          `\mathcal{E}_0`.

        - ``I2`` -- the `2\times 2` identity matrix in
          `\mathbb{M}_2(\mathcal{O}_0)`.

        - ``matrix_space`` -- the `MatrixSpace` `\mathbb{M}_2(\mathbb{H}_p)`.

        - ``IKO_Matrix`` -- a factory function to instantiate a matrix in
          `\mathbb{M}_2(\mathbb{H}_p)` from its entries.

        - ``i, ζ, ξ`` -- generators for
          `\mathcal{O}_0=\mathbb{Z}[i,\zeta,\xi]`.
          Explicitly, `\zeta=\frac{i+j}{2}` and `\xi=\frac{1+k}{2}`.

        EXAMPLES::

            sage: (iko, p, F, ω, Fx, x, Hp, O0, E0, A0, ι, π, _22_matrices, I2,
            ....:  matrix_space, IKO_Matrix, i, ζ, ξ) = IKO(6143).dump()
        """

        t = 'p,F,ω,Fx,x,Hp,O0,E0,A0,ι,π,_22_matrices,I2,matrix_space,matrix'
        names = t.split(',')
        output = (self,
                  *[getattr(self, name) for name in names],
                  *self.O0.gens()[1:])
        return output

    def ι(P: EllipticCurvePoint_finite_field
          ) -> EllipticCurvePoint_finite_field:
        r"""
        Return the automorphism `(x,y)\mapsto(-x,iy)` on `\mathcal{E}_0`,
        corresponding to `i\in\mathbb{H}_p`.
        """
        E0 = P._curve()
        ω = E0.base_ring().gen()
        if P.is_zero():
            return P
        (x, y) = P.xy()
        return E0([-x, ω * y])

    def π(P: EllipticCurvePoint_finite_field
          ) -> EllipticCurvePoint_finite_field:
        r"""
        Return the Frobenius automorphism `(x,y)\mapsto(x^p,y^p)` on
        `\mathcal{E}_0`, corresponding to `j\in\mathbb{H}_p`.
        """
        return P._curve().frobenius_isogeny()(P)

    @cached_property
    def I2(self) -> Matrix_generic_dense:
        r"""
        Return the `2\times 2` identity matrix in
        `\mathbb{M}_2(\mathcal{O}_0)`.
        """
        return self.matrix(1, 0, 0, 1)

    @cached_property
    def _22_matrices(self) -> tuple[Matrix_generic_dense]:
        r"""
        A tuple of 35 matrices in `\mathbb{M}_2(\mathcal{O}_0)`, encoding
        `(2,2)`-isogenies from `\mathcal{E}_0`.
        """
        _, i, ζ, ξ = self.O0.gens()
        return (
            self.matrix(2, 0, 0, 1),
            self.matrix(1, 0, 0, 2),
            self.matrix(1, 1, -1, 1),
            self.matrix(1, i, -1, i),
            self.matrix(1 + i, 0, 0, 1 + i),
            self.matrix(0, 1 + i, 1 + i, i),
            self.matrix(i, i - ξ, -i, i + ξ),
            self.matrix(i, 1 - ζ, -i, 1 + ζ),
            self.matrix(i, 1 + i, 1 + i, 0),
            self.matrix(1, 1 - ζ, -1, 1 + ζ),
            self.matrix(1, i - ξ, -1, i + ξ),
            self.matrix(1, ξ, -i, i + ζ),
            self.matrix(i, ξ, 1, i + ζ),
            self.matrix(1, 1 + ξ, -i, ζ),
            self.matrix(1, ζ, i, 1 + ξ),
            self.matrix(1 + ξ, -1, ζ, i),
            self.matrix(ξ, i, i + ζ, 1),
            self.matrix(ξ, -1, i + ζ, i),
            self.matrix(ζ, 1, 1 + ξ, i),
            self.matrix(1, 1 + i + ζ - ξ, i, ζ + ξ),
            self.matrix(1, ζ + ξ, -i, 1 + i + ζ - ξ),
            self.matrix(i, 1 + ζ - ξ, -1, 1 + ζ + ξ),
            self.matrix(1, i - ζ + ξ, -i, i + ζ + ξ),
            self.matrix(1 + i + ζ - ξ, -i, ζ + ξ, 1),
            self.matrix(1 + i + ζ - ξ, 1, ζ + ξ, i),
            self.matrix(i + ζ - ξ, -i, -i + ζ + ξ, 1),
            self.matrix(1 + ζ - ξ, 1, 1 + ζ + ξ, i),
            self.matrix(i + ζ, -ξ, ξ, i + ζ),
            self.matrix(i + ζ, ζ, ξ, 1 + ξ),
            self.matrix(ζ, -1 - ξ, 1 + ξ, ζ),
            self.matrix(1 + ξ, ξ, ζ, i + ζ),
            self.matrix(i + ζ + ξ, ζ + ξ, i - ζ + ξ, 1 + i - ζ + ξ),
            self.matrix(1 + ζ + ξ, 1 + i + ζ - ξ, 1 + ζ - ξ, -ζ - ξ),
            self.matrix(ζ + ξ, 1 + ζ + ξ, 1 + i + ζ - ξ, 1 + ζ - ξ),
            self.matrix(1 + i + ζ - ξ, i + ζ - ξ, ζ + ξ, -i + ζ + ξ),
        )

    def IKO_Matrix(
        self,
        *entries : Iterable[QuaternionAlgebraElement_rational_field]
    ) -> Matrix_generic_dense:
        r"""
        A factory function to instantiate a matrix in
        `\mathbb{M}_2(\mathbb{H}_p)` from its entries.
        """
        M = self.matrix_space(entries)
        M.set_immutable()
        return M

    matrix = IKO_Matrix

    @staticmethod
    def CT(α: Matrix_generic_dense) -> Matrix_generic_dense:
        r"""
        Return the conjugate transpose of a matrix in
        `\mathbb{M}_2(\mathbb{H}_p)`.
        """
        M = α.C.T
        M.set_immutable()
        return M

    @staticmethod
    def inverse(α: Matrix_generic_dense) -> Matrix_generic_dense:
        r"""
        Return the inverse of a matrix in
        `\mathbb{M}_2(\mathbb{H}_p)`.
        """
        a, b, c, d = α.list()
        unorm = (a.reduced_norm() * d.reduced_norm()
                 + b.reduced_norm() * c.reduced_norm()
                 - (a.conjugate() * b * d.conjugate() * c).reduced_trace())
        β = α.C.T
        x, y, _, z = (α * β).list()
        uinv = β * α.parent()([z, -y, -y.conjugate(), x]) / unorm
        uinv.set_immutable()
        return uinv

    @staticmethod
    def deg(φ: Matrix_generic_dense) -> Integer | Rational:
        r"""
        Return the degree (reduced norm) of a matrix in
        `\mathbb{M}_2(\mathbb{H}_p)`.
        """
        a, b, c, d = (φ * φ.C.T).list()
        degree = a * d - b * c
        if degree in ZZ:
            return ZZ(degree)
        return QQ(degree)

    @staticmethod
    def degrd(φ: Matrix_generic_dense) -> Integer | Rational:
        r"""
        Return the reduced degree (square root of the reduced norm) of a
        matrix in `\mathbb{M}_2(\mathbb{H}_p)`.

        Raise a :exc:`ValueError` if the degree is not a perfect square.
        """
        d = sqrt(IKO.deg(φ))
        if d in ZZ:
            return ZZ(d)
        if d in QQ:
            return QQ(d)
        raise ValueError("Degree is not a square.")

    def is_polarisation(self: IKO | QuaternionOrder,
                        λ: Matrix_generic_dense) -> bool:
        r"""
        Return whether the matrix ``λ`` is a polarisation matrix.
        Specifically, if `\lambda=\begin{pmatrix} a&b \\ c&d \end{pmatrix}`,
        check whether `a,d\in\mathbb{Z}^+` and `\overline{c}=b\in\mathcal{O}_0`
        and `ad-bc=1`.

        This method can be used as either an instance or class method. In the
        latter case, ``self`` is either an ``IKO`` object or the quaternion
        order `\mathcal{O}_0`.

        INPUT:

        - ``self`` -- either an ``IKO`` object or the quaternion
          order `\mathcal{O}_0`.

        - ``λ`` -- a matrix with entries in `\mathcal{O}_0`.
        """
        a, b, c, d = λ.list()
        O0 = self.O0 if isinstance(self, IKO) else self
        return (a in ZZ and a > 0
                and d in ZZ and d > 0
                and b in O0 and b.conjugate() == c
                and a * d - b * c == 1)

    @staticmethod
    def codomain(φ: Matrix_generic_dense,
                 λ: Matrix_generic_dense) -> Matrix_generic_dense:
        r"""
        Return the polarisation matrix encoding the codomain of ``φ``.

        INPUT:

        - ``φ`` -- an isogeny matrix `\varphi`.

        - ``λ`` -- polarisation matrix encoding the domain of `\varphi`.
        """
        invφ = IKO.inverse(φ)
        output = IKO.degrd(φ) * invφ.C.T * λ * invφ
        output.set_immutable()
        return output

    @staticmethod
    def map_point(
        α: QuaternionAlgebraElement_rational_field | Matrix_generic_dense,
        P: EllipticCurvePoint_finite_field | CartesianProduct.element_class
    ) -> EllipticCurvePoint_finite_field | CartesianProduct.element_class:
        r"""
        Return the image of ``P`` under the endomorphism ``α``. Works for
        points on `\mathcal{E}_0` or `\mathcal{A}_0`.

        INPUT:

        - ``α`` -- a quaternion `\alpha\in\mathcal{O}_0` or an isogeny matrix
          `\alpha\in\mathbb{M}_2(\mathcal{O}_0)`.

        - ``P`` -- a point on `\mathcal{E}_0` or `\mathcal{A}_0`.

        OUTPUT: The image `\alpha(P)`.
        """
        if isinstance(α.parent(), QuaternionAlgebra_ab):
            if α == 1 or P.is_zero():
                return P
            elif α == -1:
                return -P
            elif α == 0:
                return P._curve()(0)
            coeff = α.coefficient_tuple()
            d = α.denominator()
            Qlist = P.division_points(d)
            assert Qlist, (
                f"The {d}-division points of P={P} lie in an field extension."
            )
            # if not Qlist:
            #     FF = E.base_ring().extension(2,'ζ')
            #     P = P.change_ring(FF)
            #     Qlist = P.division_points(d)
            images = set()
            for Q in Qlist:
                Q1 = (d * coeff[0]) * Q
                Q2 = (d * coeff[1]) * IKO.ι(Q)
                Q3 = (d * coeff[2]) * IKO.π(Q)
                Q4 = (d * coeff[3]) * IKO.ι(IKO.π(Q))
                images.add(Q1 + Q2 + Q3 + Q4)
            if len(images) != 1:
                print(
                    f"[WARN] {len(images)} possible images.")
            return images.pop()
        else:
            a, b, c, d = α.list()
            return P.parent(IKO.map_point(a, P[0]) + IKO.map_point(b, P[1]),
                            IKO.map_point(c, P[0]) + IKO.map_point(d, P[1]))

    def change_basis(
        self: IKO | QuaternionOrder,
        α: Matrix_generic_dense |
           QuaternionAlgebraElement_rational_field
    ) -> (Vector_integer_dense | Vector_rational_dense |
          Matrix_integer_dense | Matrix_rational_dense):
        r"""
        Return the coordinates of an endomorphism ``α`` with
        respect to the `\mathbb{Z}`-basis of its endomorphism ring.

        This method can be used as either an instance or class method. In the
        latter case, ``self`` is either an ``IKO`` object or the quaternion
        order `\mathcal{O}_0`.

        INPUT:

        - ``self`` -- either an ``IKO`` object or the quaternion
          order `\mathcal{O}_0`.

        - ``α`` -- a quaternion `\alpha\in\mathcal{O}_0` or an isogeny matrix
          `\alpha\in\mathbb{M}_2(\mathcal{O}_0)`. 

        OUTPUT: The coordinate vector of ``\alpha`` with respect to the basis
        `(1,i,\zeta,\xi)` of `\mathcal{O}_0`, or the basis `(\mathbf{E}_{11},
        i\mathbf{E}_{11},\zeta\mathbf{E}_{11},\xi\mathbf{E}_{11},\hdots,
        \mathbf{E}_{22},i\mathbf{E}_{22},\zeta\mathbf{E}_{22},
        \xi\mathbf{E}_{22})` of `\mathbb{M}_2(\mathcal{O}_0)`.
        """

        if isinstance(α.parent(), QuaternionAlgebra_ab):
            iko = isinstance(self, IKO)
            if iko:
                G = self.O0.basis_matrix()
            else:
                G = self.basis_matrix()

            α = vector(α.coefficient_tuple())
            v = α * ~G
            if all(vi in ZZ for vi in v):
                return vector(ZZ, v)
            else:
                return v
        else:
            rows = [IKO.change_basis(self, γ) for γ in α.list()]
            G = matrix(ZZ, rows)
            if all(m in ZZ for m in G.list()):
                return matrix(ZZ, G)
            else:
                return G
