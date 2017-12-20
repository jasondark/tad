/*******************************************************************************
 * Licensed under the ISC License (ISC)                                        *
 *                                                                             *
 * Copyright 2017 Jason Dark (www.jkdark.com)                                  *
 *                                                                             *
 * Permission to use, copy, modify, and/or distribute this software for any    *
 * purpose with or without fee is hereby granted, provided that the above      *
 * copyright notice and this permission notice appear in all copies.           *
 *                                                                             *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES    *
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF            *
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY *
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES          *
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN       *
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR  *
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.                 *
 ******************************************************************************/


/*******************************************************************************
 * TAD: Templated Automatic Differentiation library                            *
 *                                                                             *
 * This is a proof-of-concept library that (ab)uses C++11 features to provide  *
 * automatic differentiation for functions specified as types. Independent     *
 * variables are enumerated as the types `Variable<0>`, `Variable<1>`, etc.    *
 *                                                                             *
 * Functions are specified via composition and derivatives via the associated  *
 * types `::derivative<0>`, `::derivative<1>`, etc.                            *        
 *                                                                             *
 * Any expression can be simplified with the associated type `::canonical`.    *
 *                                                                             *
 * For example:                                                                *
 *     using namespace tad;                                                    *
 *     double x[] = { 2.0, 5.0 };                                              *
 *                                                                             *
 *     // softmax(x) = log(1+exp(x))                                           *
 *     typedef Log<Sum<Constant<1>,Exp<Variable<0>>>>::canonical Softmax;      *
 *     cout << "log(1+exp(2)) == " << Softmax::eval(x) << endl;                *
 *                                                                             *
 *     // logistic(x) = exp(x) / (1+exp(x)) = (d/dx) softmax(x)                *
 *     typedef Softmax::derivative<0>::canonical Logistic;                     *
 *     cout << "exp(2) / (1+exp(2)) == " << Logistic::eval(x) << endl;         *
 *                                                                             *
 *     // f(x,y) = sqrt(x^2+y)                                                 *
 *     typedef Sqrt<                                                           *
 *         Sum<Power<Variable<0>,Constant<2>>,Variable<1>>                     *
 *     >::canonical F;                                                         *
 *     cout << "sqrt(2^2 + 5) == " << F::eval(x) << endl;                      *
 *                                                                             *
 * In theory, each operation is inlined. Coupled with -ffast-math, it may      *
 * permit the compiler to make some serious reductions in the evaluation of    *
 * the functions and their derivatives. The following functions are provided:  *
 *                                                                             *
 * Unary: Exp, Log, Sqrt, Cbrt,                                                *
 *        Sin,  Cos,  Tan,  Csc,  Sec,  Cot,                                   *
 *        Sinh, Cosh, Tanh, Csch, Sech, Coth,                                  *
 *        ArcSin, ArcCos, ArcTan, ArcCsc, ArcSec, ArcCot,                      *
 *        ArSinh, ArCosh, ArTanh, ArCsch, ArSech, ArCoth                       *
 *                                                                             *
 * Binary: Power<X,Y>      = x^y,                                              *
 *         Difference<X,Y> = x-y,                                              *
 *         Ratio<X,Y>      = x/y                                               *
 *                                                                             *
 * n-ary: Sum<X,Y,...>     = x+y+...                                           *
 *        Product<X,Y,...> = x*y*...                                           *
 *                                                                             *
 * Special: Constant<n> = (double) n,                                          *
 *          ConstantPI  = 3.14159...,                                          *
 *          ConstantE   = 2.71828...,                                          *
 *          Variable<i> = x[i],                                                *
 *          Delta<i,j>  = (i == j)? 1 : 0,                                     *
 *          Indeterminate = 0.0 / 0.0                                          *
 *                                                                             *
 * When possible, template specializations are provided when an argument is    *
 * Constant<0> or Constant<1> and the expression can be reduced. For example,  *
 *     Product<..., Constant<0>, ...>::canonical == Constant<0>                *
 *                                                                             *
 * Possible extension: currently, this library is restricted to double         *
 * precision numbers. A search and replace `s/double/complex/g` will change it *
 * to complex numbers. Alternatively, one could typedef or macro the scalar    *
 * type. If there is a good reason why this is useful, submit a pull request   *
 * or contact me and let me know.                                              *
 ******************************************************************************/

#ifndef TAD_H
#define TAD_H

#include <cmath>

namespace tad {







// 4 basic types that illustrate the approach of this library:
// Each function has a corresponding type, that specifies
//  * a canonical representation, and optionally
//  * the derivatives of the function and
//  * the eval() static method

// An Indeterminate type
struct Indeterminate {
    using canonical = Indeterminate;
    template <int i> using derivative = Indeterminate;
    static inline double eval(double *x) { return 0.0 / 0.0; }
};

// An integer Constant type
template <int n> struct Constant {
    using canonical = Constant<n>;
    template <int i> using derivative = Constant<0>;
    static inline double eval(double *x) { return (double) n; }
};

// Two special constants
struct ConstantPI {
    using canonical = ConstantPI;
    template <int i> using derivative = Constant<0>;
    static inline double eval(double *x) { return 3.141592653589793; }
};
struct ConstantE {
    using canonical = ConstantE;
    template <int i> using derivative = Constant<0>;
    static inline double eval(double *x) { return 2.718281828459045; }
};

// A (Kronecker-) Delta type
template <int i, int j> struct Delta      { using canonical = Constant<0>; };
template <int i>        struct Delta<i,i> { using canonical = Constant<1>; };

// An indexed Variable type
template <int i> struct Variable {
    using canonical = Variable<i>;
    template <int j> using derivative = Delta<i,j>;
    static inline double eval(double *x) { return x[i]; }
};









// Now for something a bit more advanced, we represent a Sum of functions.
// Notice that `canonical` is greedily evaluated for each type. This is mostly
// to exploit Constant<0>'s that may appear when taking derivatives.

// (empty sum)
template <class ...Empty> struct Sum {
    using canonical = Constant<0>;
};
// (singleton sum)
template <class X> struct Sum<X> {
    using canonical = typename X::canonical;
};
// (binary sum)
template <class X, class Y> struct Sum<X,Y> {
    using canonical = Sum<typename X::canonical, typename Y::canonical>;
    template <int i> using derivative =
        typename Sum<
            typename X::canonical::template derivative<i>::canonical,
            typename Y::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) { return X::canonical::eval(x) + Y::canonical::eval(x); }
};
// (binary sum) -- with specializations to skip non-contributing terms
template <class X> struct Sum<X, Constant<0>> { using canonical = typename X::canonical; };
template <class X> struct Sum<Constant<0>, X> { using canonical = typename X::canonical; };
template <> struct Sum<Constant<0>, Constant<0>> { using canonical = Constant<0>; };
// (variadic sum) -- reduce to canonical (nested binary) form
template <class X, class Y, class Z, class ...Args> struct Sum<X,Y,Z,Args...>{
    using canonical = typename Sum<
        typename Sum<typename X::canonical, typename Y::canonical>::canonical,
        typename Sum<typename Z::canonical,               Args...>::canonical
    >::canonical;
};

// The variadic Product is almost identical to Sum, except we get extra nice
// simplifications when Constant<0> appears

// (empty product)
template <class ...Empty> struct Product {
    using canonical = Constant<1>;
};
// (singleton product)
template <class X> struct Product<X> {
    using canonical = typename X::canonical;
};
// (binary product)
template <class X, class Y> struct Product<X,Y> {
    using canonical = Product<typename X::canonical, typename Y::canonical>;
    template <int i> using derivative =
        typename Sum<
            typename Product<typename X::canonical::template derivative<i>::canonical, typename Y::canonical>::canonical,
            typename Product<typename X::canonical, typename Y::canonical::template derivative<i>::canonical>::canonical>::canonical;
    static inline double eval(double *x) { return X::canonical::eval(x) * Y::canonical::eval(x); }
};
// (binary product) -- with specializations for Constant<0> and/or Constant<1>
template <class X> struct Product<X, Constant<0>> { using canonical = Constant<0>; };
template <class X> struct Product<Constant<0>, X> { using canonical = Constant<0>; };
template <class X> struct Product<X, Constant<1>> { using canonical = typename X::canonical; };
template <class X> struct Product<Constant<1>, X> { using canonical = typename X::canonical; };
template <> struct Product<Constant<0>, Constant<0>> { using canonical = Constant<0>; };
template <> struct Product<Constant<1>, Constant<0>> { using canonical = Constant<0>; };
template <> struct Product<Constant<0>, Constant<1>> { using canonical = Constant<0>; };
template <> struct Product<Constant<1>, Constant<1>> { using canonical = Constant<1>; };
// (variadic product) -- reduce to canonical binary product form
template <class X, class Y, class Z, class ...Args> struct Product<X,Y,Z,Args...>{
    using canonical = 
        typename Product<
            typename Product<typename X::canonical, typename Y::canonical>::canonical,
            typename Product<typename Z::canonical,               Args...>::canonical>::canonical;
};









// Now for some binary functions: Difference, Ratio

// Representing a Difference with a specialization for Difference<X,X> = Constant<0>
template <class X, class Y> struct Difference {
    using canonical = typename Sum<
        typename X::canonical,
        typename Product<
            Constant<-1>,
            typename Y::canonical>::canonical>::canonical;
};
template <class X> struct Difference<X,X> { using canonical = Constant<0>; };

// Representing a Ratio -- note the use of the quotient rule to construct the derivative
template <class X, class Y> struct Power;
template <class X, class Y> struct Ratio {
    using canonical = Ratio<typename X::canonical, typename Y::canonical>;
    template <int i> using derivative =
        typename Ratio<
            typename Difference<
                typename Product<typename X::canonical::template derivative<i>::canonical, typename Y::canonical>::canonical,
                typename Product<typename X::canonical, typename Y::canonical::template derivative<i>::canonical>::canonical>::canonical,
            typename Power<typename X::canonical, Constant<2>>::canonical>::canonical;
    static inline double eval(double *x) { return X::canonical::eval(x) / Y::canonical::eval(x); }
};
// and specializations for X/X, 0/X, and 0/0
template <class X> struct Ratio<X,X>               { using canonical = Constant<1>;   };
template <class X> struct Ratio<Constant<0>, X>    { using canonical = Constant<0>;   };
template <> struct Ratio<Constant<0>, Constant<0>> { using canonical = Indeterminate; };









// Now a basic Exp and Log implementation

// Exp, with a specialization for Exp<Constant<0>>
template <class X> struct Exp {
    using canonical = Exp<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            typename Exp<typename X::canonical>::canonical,
            typename X::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) { return exp(X::canonical::eval(x)); }
};
template <> struct Exp<Constant<0>> { using canonical = Constant<1>; };

// Log, with a specialization for Log<Constant<1>>
template <class X> struct Log {
    using canonical = Log<typename X::canonical>;
    template <int i> using derivative = typename Ratio<typename X::canonical::template derivative<i>::canonical, typename X::canonical>::canonical;
    static inline double eval(double *x) { return log(X::canonical::eval(x)); }
};
template <> struct Log<Constant<1>> { using canonical = Constant<0>; };









// Finally, a general implementation of Power -- note that we express the derivative using x^y = exp(y*log(x))
template <class X, class Y> struct Power {
    using canonical = Power<typename X::canonical, typename Y::canonical>;
    template <int i> using derivative =
        typename Exp<
            typename Product<
                typename Y::canonical,
                typename Log<typename X::canonical>::canonical>::canonical>::canonical::template derivative<i>::canonical;
    static inline double eval(double *x) { return pow(X::canonical::eval(x), Y::canonical::eval(x)); }
};
// with a few specific specializations
template <class X> struct Power<X, Constant<-1>> { using canonical = typename Ratio<Constant<1>, typename X::canonical>::canonical; };
template <class X> struct Power<X, Constant< 0>> { using canonical = Constant<1>; };
template <class X> struct Power<X, Constant< 1>> { using canonical = typename X::canonical; };
template <class X> struct Power<X, Constant< 2>> {
    using canonical = Power<typename X::canonical, Constant<2>>;
    template <int i> using derivative = typename Product<Constant<2>, typename X::canonical, typename X::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) {
        double y = X::canonical::eval(x);
        return y*y;
    }
};









// Straight-forward, but messy, implementations of Sqrt and Cbrt with specializations for Constant<0>

template <class X> struct Sqrt {
    using canonical = Sqrt<typename X::canonical>;
    template <int i> using derivative = typename Product<
        Ratio<Constant<1>,Constant<2>>,
        typename Ratio<
            typename X::canonical::template derivative<i>::canonical,
            typename Sqrt<typename X::canonical>::canonical
        >::canonical
    >::canonical;
    static inline double eval(double *x) { return sqrt(X::canonical::eval(x)); }
};
template <> struct Sqrt<Constant<0>> { using canonical = Constant<0>; };

template <class X> struct Cbrt {
    using canonical = Cbrt<typename X::canonical>;
    template <int i> using derivative = typename Product<
        Ratio<Constant<1>,Constant<3>>,
        typename Ratio<
            typename X::canonical::template derivative<i>::canonical,
            typename Power<
                typename Cbrt<typename X::canonical>::canonical,
                Constant<2>
            >::canonical
        >::canonical
    >::canonical;
    static inline double eval(double *x) { return cbrt(X::canonical::eval(x)); }
};
template <> struct Cbrt<Constant<0>> { using canonical = Constant<0>; };









// Trig and Hyperbolic functions
template <class X> struct Cos;
template <class X> struct Sin {
    using canonical = Sin<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            typename Cos<typename X::canonical>::canonical,
            typename X::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) { return sin(X::canonical::eval(x)); }
};
template <class X> struct Cos {
    using canonical = Cos<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            Constant<-1>,
            typename Sin<typename X::canonical>::canonical,
            typename X::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) { return cos(X::canonical::eval(x)); }
};
template <class X> struct Tan {
    using canonical = Tan<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            typename Power<
                typename Cos<typename X::canonical>::canonical,
                Constant<-2>>::canonical,
            typename X::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) { return tan(X::canonical::eval(x)); }
};
template <class X> struct Cosh;
template <class X> struct Sinh {
    using canonical = Sinh<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            typename Cosh<typename X::canonical>::canonical,
            typename X::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) { return sinh(X::canonical::eval(x)); }
};
template <class X> struct Cosh {
    using canonical = Cosh<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            typename Sinh<typename X::canonical>::canonical,
            typename X::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) { return cosh(X::canonical::eval(x)); }
};
template <class X> struct Tanh {
    using canonical = Tanh<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            typename Power<
                typename Cosh<typename X::canonical>::canonical,
                Constant<-2>>::canonical,
            typename X::canonical::template derivative<i>::canonical>::canonical;
    static inline double eval(double *x) { return tanh(X::canonical::eval(x)); }
};

template<> struct Sin< Constant<0>> { using canonical = Constant<0>; };
template<> struct Cos< Constant<0>> { using canonical = Constant<1>; };
template<> struct Tan< Constant<0>> { using canonical = Constant<0>; };
template<> struct Sinh<Constant<0>> { using canonical = Constant<0>; };
template<> struct Cosh<Constant<0>> { using canonical = Constant<1>; };
template<> struct Tanh<Constant<0>> { using canonical = Constant<0>; };

template <class X> using Csc  = typename Ratio<Constant<1>, typename  Sin<typename X::canonical>::canonical>::canonical;
template <class X> using Sec  = typename Ratio<Constant<1>, typename  Cos<typename X::canonical>::canonical>::canonical;
template <class X> using Cot  = typename Ratio<Constant<1>, typename  Tan<typename X::canonical>::canonical>::canonical;
template <class X> using Csch = typename Ratio<Constant<1>, typename Sinh<typename X::canonical>::canonical>::canonical;
template <class X> using Sech = typename Ratio<Constant<1>, typename Cosh<typename X::canonical>::canonical>::canonical;
template <class X> using Coth = typename Ratio<Constant<1>, typename Tanh<typename X::canonical>::canonical>::canonical;









// Arc and Area (inverse) trig functions
template <class X> struct ArcSin {
    using canonical = ArcSin<typename X::canonical>;
    template <int i> using derivative =
        typename Ratio<
            Constant<1>,
            typename Sqrt<
                typename Difference<
                    Constant<1>,
                    typename Power<typename X::canonical, Constant<2>>::canonical
                >::canonical
            >::canonical
        >::canonical;
    static inline double eval(double *x) { return asin(X::canonical::eval(x)); }
};
template <class X> struct ArcCos {
    using canonical = ArcCos<typename X::canonical>;
    template <int i> using derivative =
        typename Ratio<
            Constant<-1>,
            typename Sqrt<
                typename Difference<
                    Constant<1>,
                    typename Power<typename X::canonical, Constant<2>>::canonical
                >::canonical
            >::canonical
        >::canonical;
    static inline double eval(double *x) { return acos(X::canonical::eval(x)); }
};
template <class X> struct ArcTan {
    using canonical = ArcTan<typename X::canonical>;
    template <int i> using derivative =
        typename Ratio<
            Constant<1>,
            typename Sum<
                Constant<1>,
                typename Power<typename X::canonical, Constant<2>>::canonical
            >::canonical
        >::canonical;
    static inline double eval(double *x) { return atan(X::canonical::eval(x)); }
};
template <class X> struct ArSinh {
    using canonical = ArSinh<typename X::canonical>;
    template <int i> using derivative =
        typename Ratio<
            Constant<1>,
            typename Sqrt<
                typename Sum<
                    Constant<1>,
                    typename Power<typename X::canonical, Constant<2>>::canonical
                >::canonical
            >::canonical
        >::canonical;
    static inline double eval(double *x) { return asinh(X::canonical::eval(x)); }
};
template <class X> struct ArCosh {
    using canonical = ArCosh<typename X::canonical>;
    template <int i> using derivative =
        typename Ratio<
            Constant<1>,
            typename Sqrt<
                typename Sum<
                    Constant<-1>,
                    typename Power<typename X::canonical, Constant<2>>::canonical
                >::canonical
            >::canonical
        >::canonical;
    static inline double eval(double *x) { return acosh(X::canonical::eval(x)); }
};
template <class X> struct ArTanh {
    using canonical = ArTanh<typename X::canonical>;
    template <int i> using derivative =
        typename Ratio<
            Constant<1>,
            typename Difference<
                Constant<1>,
                typename Power<typename X::canonical, Constant<2>>::canonical
            >::canonical
        >::canonical;
    static inline double eval(double *x) { return atanh(X::canonical::eval(x)); }
};

template<> struct ArcSin<Constant<0>> { using canonical = Constant<0>; };
template<> struct ArcCos<Constant<1>> { using canonical = Constant<0>; };
template<> struct ArcTan<Constant<0>> { using canonical = Constant<0>; };
template<> struct ArSinh<Constant<0>> { using canonical = Constant<0>; };
template<> struct ArCosh<Constant<1>> { using canonical = Constant<0>; };
template<> struct ArTanh<Constant<0>> { using canonical = Constant<0>; };

template <class X> using ArcCsc = typename ArcSin<typename Ratio<Constant<1>, typename X::canonical>::canonical>::canonical;
template <class X> using ArcSec = typename ArcCos<typename Ratio<Constant<1>, typename X::canonical>::canonical>::canonical;
template <class X> using ArcCot = typename ArcTan<typename Ratio<Constant<1>, typename X::canonical>::canonical>::canonical;
template <class X> using ArCsch = typename ArSinh<typename Ratio<Constant<1>, typename X::canonical>::canonical>::canonical;
template <class X> using ArSech = typename ArCosh<typename Ratio<Constant<1>, typename X::canonical>::canonical>::canonical;
template <class X> using ArCoth = typename ArTanh<typename Ratio<Constant<1>, typename X::canonical>::canonical>::canonical;









// The error function and its complement

template <class X> struct Erf {
    using canonical = Erf<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            Ratio<Constant<2>,Sqrt<ConstantPI>>,
            typename Exp<typename Power<typename X::canonical, Constant<-2>>::canonical>::canonical>::canonical;
    static inline double eval(double *x) { return erf(X::canonical::eval(x)); }
};
template <class X> struct Erfc {
    using canonical = Erfc<typename X::canonical>;
    template <int i> using derivative =
        typename Product<
            Ratio<Constant<-2>,Sqrt<ConstantPI>>,
            typename Exp<typename Power<typename X::canonical, Constant<-2>>::canonical>::canonical>::canonical;
    static inline double eval(double *x) { return erfc(X::canonical::eval(x)); }
};









// Sadly, no gamma functions -- their derivatives require additional special functions
// not available in <cmath>

// Also, this is it. No more smooth functions to be had in <cmath>

} // end namespace declaration

#endif

