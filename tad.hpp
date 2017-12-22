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
 * Functions are specified via composition and derivatives computed at         *
 * compile-time via the `Derivative<F<X>,X>`.                                  *
 *                                                                             *
 * Any expression can be simplified with the associated type `::canonical`.    *
 *                                                                             *
 * For example:                                                                *
 *     template <class X>                                                      *
 *     using Softmax = typename Log<Sum<One,Exp<X>>>::canonical;               *
 *     template<class X>                                                       *
 *     using Logistic = Derivative<Softmax<X>, X>;                             *
 *                                                                             *
 *     // ...                                                                  *
 *                                                                             *
 *     typedef Variable<0> X;                                                  *
 *     typedef Variable<1> Y;                                                  *
 *                                                                             *
 *     double x[] = { 2.0, 5.0 };                                              *
 *     cout << "log(1+exp(2)) == "       << Softmax<X>::eval(x)  << endl;      *
 *     cout << "exp(5) / (1+exp(5)) == " << Logistic<Y>::eval(x) << endl;      *
 *                                                                             *
 *     // use complex numbers! assuming #include <complex> is present...       *
 *     std::complex<double> z[] = { 0.5 };                                     *
 *     cout << "ArCosh(0.5) == " << ArCosh<X>::eval(z) << endl;                *
 *     // any numeric type that overloads <cmath> functions will work          *
 *                                                                             *
 *     // inline specification of sqrt(x^2+y)                                  *
 *     typedef Sqrt<Sum<Power<X,Constant<2>>,Y>>::canonical Expr;              *
 *     cout << "sqrt(2^2 + 5) == " << Expr::eval(x) << endl;                   *
 *     // Expr's derivatives can be obtained directly:                         *                    
 *     // typedef Derivative<Expr,X> Expr_dx;                                  *
 *     // typedef Derivative<Expr,Y> Expr_dy;                                  *
 *                                                                             *
 * In theory, each operation is inlined. Coupled with -ffast-math, it may      *
 * permit the compiler to make some serious reductions in the evaluation of    *
 * the functions and their derivatives. The following functions are provided:  *
 *                                                                             *
 * Unary: Negative, Reciprocal, Square, Exp, Log, Sqrt, Cbrt,                  *
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
 * Special: Zero, One, Two, EulerE, Three, Pi                                  *
 *          Variable<i> = x[i],                                                *
 *          Delta<X,Y>  = (X == Y)? 1 : 0,                                     *
 *          Indeterminate = 0.0 / 0.0,                                         *
 *          Derivative<Expression,Variable>,                                   *
 *          Canonical<X,Y> (internal use only)                                 *
 *                                                                             *
 * When possible, template specializations are provided when an argument is    *
 * Zero or One and the expression can be reduced. For example,                 *
 *     Product<..., Zero, ...>::canonical == Zero                              *
 * This is why calling `::canonical` on a user-constructed function is         *
 * potentially useful. (::canonical is automatically expanded for any internal *
 * composition.)                                                               *
 *                                                                             *
 * In addition to scalar operations, support for an Array<...> type has been   *
 * added:                                                                      *
 *     typedef Array<Variable<0>, Variable<1>, Variable<2>>::canonical U;      *
 *     typedef Array<Variable<3>, Variable<4>, Variable<5>>::canonical V;      *
 *     typedef DotProduct<                                                     *
 *         Map<Sum, Map<Square,U>, Map<Square<V>>,                             *
 *         U>::canonical Result;                                               *
 *     cout << Result::eval(x) << endl;                                        *
 *                                                                             *
 * As one can see, Array<> takes a list of types, DotProduct<X,Y> computes     *
 * the dot product of equally-sized Array's X and Y, and Map can apply unary   *
 * and binary operations element-wise. If eval() is called on an Array type,   *
 * a std::array<scalar_t, N> is returned, rather than scalar_t. Derivatives    *
 * thread naturally over an Array, as to be expected.                          *
 *                                                                             *
 ******************************************************************************/

#ifndef TAD_H
#define TAD_H

#include <cmath>         // the scalar operations call standard library functions
#include <array>         // the return type of an Array expression
#include <type_traits>   // used for static assertions to ensure canonicalization

namespace tad {
using std::array;

// an internal "method" to recursively enforce the canonical form
// (necessary since derivative and eval() are not always defined for
// non-canonical forms)
template <class X, class Y> struct Canonical      { using type = typename Canonical<Y, typename Y::canonical>::type; };
template <class X>          struct Canonical<X,X> { using type = X; };


// The derivative -- an ergonomic wrapper over the internally implemented
// template derivatives with automatic canonicalization
template <class F, class X>
using Derivative = typename F::canonical::template derivative<X>::canonical;


// Some basic types that illustrate the approach of this library:
// Each function has a corresponding type, that specifies
//  * a canonical representation, and optionally
//  * the derivatives of the function and
//  * the eval() static method

// An Indeterminate type -- used for Ratio<0,0> (should probably never appear in the wild)
struct Indeterminate {
    using canonical = Indeterminate;
    template <class Z> using derivative = Indeterminate;
    template <class scalar> static inline scalar eval(const scalar *x) { return 0.0 / 0.0; }
};

// The Zero type (useful for simplifying derivatives and products)
struct Zero {
    using canonical = Zero;
    template <class Z> using derivative = Zero;
    template <class scalar> static inline scalar eval(const scalar *x) { return 0.0; }
};
// The One type, used along with the zero type for the kronecker-delta
struct One {
    using canonical = One;
    template <class Z> using derivative = Zero;
    template <class scalar> static inline scalar eval(const scalar *x) { return 1.0; }
};
// The remaining constants are arbitrary and are implemented to overcome the inability to encode
// non-integers in the type system
struct Two {
    using canonical = Two;
    template <class Z> using derivative = Zero;
    template <class scalar> static inline scalar eval(const scalar *x) { return 2.0; }
};
struct Three {
    using canonical = Three;
    template <class Z> using derivative = Zero;
    template <class scalar> static inline scalar eval(const scalar *x) { return 3.0; }
};
struct Pi {
    using canonical = Pi;
    template <class Z> using derivative = Zero;
    template <class scalar> static inline scalar eval(const scalar *x) { return 3.141592653589793; }
};
struct EulerE {
    using canonical = EulerE;
    template <class Z> using derivative = Zero;
    template <class scalar> static inline scalar eval(const scalar *x) { return 2.718281828459045; }
};

// A (Kronecker-) Delta type
template <class X, class Y> struct Delta      { using canonical = Zero; };
template <class X>          struct Delta<X,X> { using canonical = One; };

// An indexed Variable type
template <int i> struct Variable {
    using canonical = Variable<i>;
    template <class X> using derivative = Delta<Variable<i>,X>;
    template <class scalar> static inline scalar eval(const scalar *x) { return x[i]; }
};







// Next we implement the Sum and Product. Rather than use template packing/unpacking,
// we elect to canonicalize on a recursive binary expression, e.g. Car/Cdr/Cons representation.
// This probably hurts the compilation time, but it permits an easy expression of the product rule.
// That is, rather than template-metaprogram (f*g*h*...)' = f'*g*h*... + f*g'*h*... + f*g*h'*... + ...,
// we need only code (f*g)' = f'*g + f*g', where g represents the rest of the product. For symmetry,
// Sum is implemented the same way. We ask the compiler to inline each eval() call, in the hopes that
// it is able to generate and/or optimize away the type abstract.


// (empty sum)
template <class ...Empty> struct Sum {
    using canonical = Zero;
};
// (singleton sum)
template <class X> struct Sum<X> {
    using canonical = typename X::canonical;
};
// (binary sum)
template <class X, class Y> struct Sum<X,Y> {
    using canonical = typename Canonical<Sum<X,Y>, Sum<typename X::canonical, typename Y::canonical>>::type;
    template <class Z> using derivative = Sum<Derivative<X,Z>, Derivative<Y,Z>>;
    template <class scalar> static inline scalar eval(const scalar *x) {
        static_assert(std::is_same<Sum<X,Y>, canonical>::value, "cannot eval non-canonical representation");
        return X::eval(x) + Y::eval(x);
    }
};
// (binary sum) -- with specializations to skip non-contributing terms
template <class X> struct Sum<X, Zero>    { using canonical = typename X::canonical; };
template <class X> struct Sum<Zero, X>    { using canonical = typename X::canonical; };
template <>        struct Sum<Zero, Zero> { using canonical = Zero; };
// (variadic sum) -- reduce to canonical (nested binary) form
template <class X, class Y, class Z, class ...Args> struct Sum<X,Y,Z,Args...>{
    using canonical = typename Sum<Sum<X,Y>,Sum<Z,Args...>>::canonical;
};

// (empty product)
template <class ...Empty> struct Product {
    using canonical = One;
};
// (singleton product)
template <class X> struct Product<X> {
    using canonical = typename X::canonical;
};
// (binary product)
template <class X, class Y> struct Product<X,Y> {
    using canonical = typename Canonical<Product<X,Y>, Product<typename X::canonical, typename Y::canonical>>::type;
    template <class Z> using derivative = Sum<Product<Derivative<X,Z>, Y>, Product<X, Derivative<Y,Z>>>;
    template <class scalar> static inline scalar eval(const scalar *x) {
        static_assert(std::is_same<Product<X,Y>, canonical>::value, "cannot eval non-canonical representation");
        return X::canonical::eval(x) * Y::canonical::eval(x);
    }
};
// (binary product) -- with specializations for Zero and/or One
template <class X> struct Product<X, Zero> { using canonical = Zero; };
template <class X> struct Product<Zero, X> { using canonical = Zero; };
template <class X> struct Product<X, One>  { using canonical = typename X::canonical; };
template <class X> struct Product<One, X>  { using canonical = typename X::canonical; };
template <> struct Product<Zero, Zero> { using canonical = Zero; };
template <> struct Product<One,  Zero> { using canonical = Zero; };
template <> struct Product<Zero,  One> { using canonical = Zero; };
template <> struct Product<One,   One> { using canonical = One; };
// (variadic product) -- reduce to canonical binary product form
template <class X, class Y, class Z, class ...Args> struct Product<X,Y,Z,Args...>{
    using canonical = typename Product<Product<X, Y>, Product<Z, Args...>>::canonical;
};










// Now for some useful helper functions: Square, Negative, Reciprocal, Difference, Ratio
// This is a good section for inspiration if you want to implement your own type functions.

template <class X> struct Negative {
    using canonical = typename Canonical<Negative<X>, Negative<typename X::canonical>>::type;
    template <class Z> using derivative = Negative<Derivative<X,Z>>;
    template <class scalar> static inline scalar eval(const scalar *x) {
       static_assert(std::is_same<Negative<X>, canonical>::value, "cannot eval non-canonical representation");
       return -X::eval(x);
    }
};
template <> struct Negative<Zero> { using canonical = Zero; };

template <class X> struct Square {
    using canonical = typename Canonical<Square<X>, Square<typename X::canonical>>::type;
    template <class Z> using derivative = Product<Two, X, Derivative<X,Z>>;
    template <class scalar> static inline scalar eval(const scalar *x) {
        static_assert(std::is_same<Square<X>, canonical>::value, "cannot eval non-canonical representation");
        double y = X::eval(x);
        return y*y;
    }
};
template <> struct Square<Zero> { using canonical = Zero; };
template <> struct Square<One>  { using canonical = One;  };

// Representing a Difference with a specialization for Difference<X,X> = Zero
template <class X, class Y> using Difference = typename Sum<X,Negative<Y>>::canonical;
template <class X> struct Sum<X,Negative<X>> { using canonical = Zero; };

// Representing a Ratio -- note the use of the quotient rule to construct the derivative
template <class X, class Y> struct Ratio {
    using canonical = typename Canonical<Ratio<X,Y>, Ratio<typename X::canonical, typename Y::canonical>>::type;
    template <class Z> using derivative =
        Ratio<
            Difference<
                Product<Derivative<X,Z>, Y>,
                Product<X, Derivative<Y,Z>>>,
            Square<Y>>;
    template <class scalar> static inline scalar eval(const scalar *x) {
        static_assert(std::is_same<Ratio<X,Y>, canonical>::value, "cannot eval non-canonical representation");
        return X::eval(x) / Y::eval(x);
    }
};
// and specializations for X/X, 0/X, and 0/0
template <class X> struct Ratio<X,X>        { using canonical = One;   };
template <class X> struct Ratio<Zero, X>    { using canonical = Zero;   };
template <>        struct Ratio<Zero, Zero> { using canonical = Indeterminate; };

template <class X> using Reciprocal = typename Ratio<One,X>::canonical;






// Now a basic Exp and Log implementation

// Exp, with a specialization for Exp<Zero>
template <class X> struct Exp {
    using canonical = typename Canonical<Exp<X>, Exp<typename X::canonical>>::type;
    template <class Z> using derivative = Product<canonical, Derivative<X,Z>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return exp(X::canonical::eval(x)); }
};
template <> struct Exp<Zero> { using canonical = One; };

// Log, with a specialization for Log<One>
template <class X> struct Log {
    using canonical = typename Canonical<Log<X>, Log<typename X::canonical>>::type;
    template <class Z> using derivative = Ratio<Derivative<X,Z>, X>;
    template <class scalar> static inline scalar eval(const scalar *x) { return log(X::canonical::eval(x)); }
};
template <> struct Log<One> { using canonical = Zero; };









// Finally, a general implementation of Power -- note that we express the derivative using x^y = exp(y*log(x))
template <class X, class Y> struct Power {
    using canonical = typename Canonical<Power<X,Y>, Power<typename X::canonical, typename Y::canonical>>::type;
    template <class Z> using derivative = Derivative<Exp<Product<Y,Log<X>>>,Z>;
    template <class scalar> static inline scalar eval(const scalar *x) { return pow(X::canonical::eval(x), Y::canonical::eval(x)); }
};
// with a few specific specializations
template <class X> struct Power<X, Zero> { using canonical = One; };
template <class X> struct Power<X, One>  { using canonical = typename X::canonical; };
template <class X> struct Power<X, Two>  { using canonical = typename Square<X>::canonical; };









// Straight-forward, but messy, implementations of Sqrt and Cbrt with specializations for Zero

template <class X> struct Sqrt {
    using canonical = typename Canonical<Sqrt<X>, Sqrt<typename X::canonical>>::type;
    template <class Z> using derivative = Product<Ratio<One,Two>, Ratio<Derivative<X,Z>, canonical>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return sqrt(X::canonical::eval(x)); }
};
template <> struct Sqrt<Zero> { using canonical = Zero; };

template <class X> struct Cbrt {
    using canonical = typename Canonical<Cbrt<X>, Cbrt<typename X::canonical>>::type;
    template <class Z> using derivative = Product<
        Ratio<One,Three>,
        Ratio<Derivative<X,Z>, Square<Cbrt<X>>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return cbrt(X::canonical::eval(x)); }
};
template <> struct Cbrt<Zero> { using canonical = Zero; };





// The next (large) section of code implements trig and hyperbolic functions and their inverses
// It is very repetitive, not very informative, and completely straight-forward.

template <class X> struct Cos; // forward declaration for Sin's derivative
template <class X> struct Sin {
    using canonical = typename Canonical<Sin<X>, Sin<typename X::canonical>>::type;
    template <class Z> using derivative = Product<Cos<X>, Derivative<X,Z>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return sin(X::canonical::eval(x)); }
};
template <class X> struct Cos {
    using canonical = typename Canonical<Cos<X>, Cos<typename X::canonical>>::type;
    template <class Z> using derivative = Negative<Product<Sin<X>, Derivative<X,Z>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return cos(X::canonical::eval(x)); }
};
template <class X> struct Tan {
    using canonical = typename Canonical<Tan<X>, Tan<typename X::canonical>>::type;
    template <class Z> using derivative = Ratio<Derivative<X,Z>, Square<Cos<X>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return tan(X::canonical::eval(x)); }
};
template <class X> struct Cosh; // forward declaration for Sinh's derivative
template <class X> struct Sinh {
    using canonical = typename Canonical<Sinh<X>, Sinh<typename X::canonical>>::type;
    template <class Z> using derivative = Product<Cosh<X>, Derivative<X,Z>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return sinh(X::canonical::eval(x)); }
};
template <class X> struct Cosh {
    using canonical = typename Canonical<Cosh<X>, Cosh<typename X::canonical>>::type;
    template <class Z> using derivative = Product<Sinh<X>, Derivative<X,Z>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return cosh(X::canonical::eval(x)); }
};
template <class X> struct Tanh {
    using canonical = typename Canonical<Tanh<X>, Tanh<typename X::canonical>>::type;
    template <class Z> using derivative = Ratio<Derivative<X,Z>, Square<Cosh<X>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return tanh(X::canonical::eval(x)); }
};

template<> struct  Sin<Zero> { using canonical = Zero; };
template<> struct  Cos<Zero> { using canonical = One; };
template<> struct  Tan<Zero> { using canonical = Zero; };
template<> struct Sinh<Zero> { using canonical = Zero; };
template<> struct Cosh<Zero> { using canonical = One; };
template<> struct Tanh<Zero> { using canonical = Zero; };

template <class X> using Csc  = typename Ratio<One,  Sin<X>>::canonical;
template <class X> using Sec  = typename Ratio<One,  Cos<X>>::canonical;
template <class X> using Cot  = typename Ratio<One,  Tan<X>>::canonical;
template <class X> using Csch = typename Ratio<One, Sinh<X>>::canonical;
template <class X> using Sech = typename Ratio<One, Cosh<X>>::canonical;
template <class X> using Coth = typename Ratio<One, Tanh<X>>::canonical;

template <class X> struct ArcSin {
    using canonical = typename Canonical<ArcSin<X>, ArcSin<typename X::canonical>>::type;
    template <class Z> using derivative = Ratio<Derivative<X,Z>, Sqrt<Difference<One,Square<X>>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return asin(X::canonical::eval(x)); }
};
template <class X> struct ArcCos {
    using canonical = typename Canonical<ArcCos<X>, ArcCos<typename X::canonical>>::type;
    template <class Z> using derivative = Negative<Ratio<Derivative<X,Z>, Sqrt<Difference<One,Square<X>>>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return acos(X::canonical::eval(x)); }
};
template <class X> struct ArcTan {
    using canonical = typename Canonical<ArcTan<X>, ArcTan<typename X::canonical>>::type;
    template <class Z> using derivative = Ratio<Derivative<X,Z>,Sum<One,Square<X>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return atan(X::canonical::eval(x)); }
};

template <class X> struct ArSinh {
    using canonical = typename Canonical<ArSinh<X>, ArSinh<typename X::canonical>>::type;
    template <class Z> using derivative = Ratio<Derivative<X,Z>, Sqrt<Sum<Square<X>,One>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return asinh(X::canonical::eval(x)); }
};
template <class X> struct ArCosh {
    using canonical = typename Canonical<ArCosh<X>, ArCosh<typename X::canonical>>::type;
    template <class Z> using derivative = Ratio<Derivative<X,Z>, Sqrt<Difference<Square<X>,One>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return acosh(X::canonical::eval(x)); }
};
template <class X> struct ArTanh {
    using canonical = typename Canonical<ArTanh<X>, ArTanh<typename X::canonical>>::type;
    template <class Z> using derivative = Ratio<One, Difference<One,Square<X>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return atanh(X::canonical::eval(x)); }
};

template<> struct ArcSin<Zero> { using canonical = Zero; };
template<> struct ArcCos<One>  { using canonical = Zero; };
template<> struct ArcTan<Zero> { using canonical = Zero; };
template<> struct ArSinh<Zero> { using canonical = Zero; };
template<> struct ArCosh<One>  { using canonical = Zero; };
template<> struct ArTanh<Zero> { using canonical = Zero; };

template <class X> using ArcCsc = typename ArcSin<Ratio<One,X>>::canonical;
template <class X> using ArcSec = typename ArcCos<Ratio<One,X>>::canonical;
template <class X> using ArcCot = typename ArcTan<Ratio<One,X>>::canonical;
template <class X> using ArCsch = typename ArSinh<Ratio<One,X>>::canonical;
template <class X> using ArSech = typename ArCosh<Ratio<One,X>>::canonical;
template <class X> using ArCoth = typename ArTanh<Ratio<One,X>>::canonical;









// The error function and its complement
template <class X> struct Erf {
    using canonical = typename Canonical<Erf<X>, Erf<typename X::canonical>>::type;
    template <class Z> using derivative =
        Product<
            Ratio<Two,Sqrt<Pi>>,
            Ratio<Derivative<X,Z>, Exp<Square<X>>>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return erf(X::canonical::eval(x)); }
};
template <class X> struct Erfc {
    using canonical = typename Canonical<Erfc<X>, Erfc<typename X::canonical>>::type;
    template <class Z> using derivative = Negative<Derivative<Erf<X>,Z>>;
    template <class scalar> static inline scalar eval(const scalar *x) { return erfc(X::canonical::eval(x)); }
};







// Finally, the Array implementation. Unlike Sum/Product, we rely on parameter packing

template <class ...X> struct Array {
	using canonical = Array<typename X::canonical...>;
	template <class Z> using derivative = Array<Derivative<X,Z>...>;

	template <class scalar>
	static inline array<scalar, sizeof...(X)> eval(const scalar *x) {
		return { X::canonical::eval(x)... };
	}
};

// Base definition: unimplemented
template <template<class...> class F, class ...X> struct Map;
// For three or more Array's, we actually implement Map-Reduce
template <template<class...> class F, class ...X, class ...Y, class ...Z, class ...Args>
struct Map<F, Array<X...>, Array<Y...>, Array<Z...>, Args...> {
	using canonical = typename Map<F,
	      typename Map<F, Array<X...>, Array<Y...>>::canonical,
	      typename Map<F, Array<Z...>, Args...>::canonical
	>::canonical;
};
// For two Array's, we get what we expect
template <template<class...> class F, class ...X, class ...Y>
struct Map<F, Array<X...>, Array<Y...>> {
	using canonical = typename Array<F<X,Y>...>::canonical;
};
// Also for one Array
template <template<class...> class F, class ...X>
struct Map<F, Array<X...>> {
	using canonical = typename Array<F<X>...>::canonical;
};

// DotProduct -- take element-wise products, then sum them
template <class X, class Y> struct DotProduct;
template <class ...X, class ...Y> struct DotProduct<Array<X...>,Array<Y...>> {
	using canonical = typename Sum<Product<X,Y>...>::canonical;
};

// TODO: convenience method to pack all the Variable's into an Array
// TODO: functions for matrix packing and operations, e.g. Array<Array<...>...>


} // end namespace declaration

#endif

