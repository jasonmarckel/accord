﻿// Accord Statistics Library
// The Accord.NET Framework
// http://accord-framework.net
//
// Copyright © César Souza, 2009-2017
// cesarsouza at gmail.com
//
//    This library is free software; you can redistribute it and/or
//    modify it under the terms of the GNU Lesser General Public
//    License as published by the Free Software Foundation; either
//    version 2.1 of the License, or (at your option) any later version.
//
//    This library is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//    Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with this library; if not, write to the Free Software
//    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
//

namespace Accord.Statistics.Models.Regression.Fitting
{
    using System;
    using Accord.Math.Optimization;
    using Accord.MachineLearning;
    using Accord.Math;

    using System.Threading;

    /// <summary>
    ///   Non-linear Least Squares for <see cref="NonlinearRegression"/> optimization.
    /// </summary>
    /// 
    /// <example>
    /// <para>
    ///   The first example shows how to fit a non-linear least squares problem with <see cref="LevenbergMarquardt"/>.</para>
    /// <code source="Unit Tests\Accord.Tests.Statistics\Models\Regression\NonlinearLeastSquaresTest.cs" region="doc_learn_lm" lang="cs"/>
    /// <code source="Unit Tests\Accord.Tests.Statistics.VB\Models\Regression\NonlinearLeastSquaresTest.vb" region="doc_learn_lm" lang="vb"/>
    /// 
    /// <para>
    ///   The second example shows how to fit a non-linear least squares problem with <see cref="GaussNewton"/>.</para>
    /// <code source="Unit Tests\Accord.Tests.Statistics\Models\Regression\NonlinearLeastSquaresTest.cs" region="doc_learn_gn" lang="cs"/>
    /// <code source="Unit Tests\Accord.Tests.Statistics.VB\Models\Regression\NonlinearLeastSquaresTest.vb" region="doc_learn_gn" lang="vb"/>
    /// </example>
    /// 
#pragma warning disable 612, 618
    public class NonlinearLeastSquares : ISupervisedLearning<NonlinearRegression, double[], double>
#pragma warning restore 612, 618
    {
        [NonSerialized]
        CancellationToken token = new CancellationToken();

        private ILeastSquaresMethod solver;
        private NonlinearRegression regression;
        private bool computeStandardErrors = true;
        private int numberOfParameters;

        RegressionFunction function;
        RegressionGradientFunction gradient;


        /// <summary>
        ///   Gets or sets a value indicating whether standard
        ///   errors should be computed in the next iteration.
        /// </summary>
        /// <value>
        /// 	<c>true</c> to compute standard errors; otherwise, <c>false</c>.
        /// </value>
        /// 
        public bool ComputeStandardErrors
        {
            get { return computeStandardErrors; }
            set { computeStandardErrors = value; }
        }

        /// <summary>
        ///   Gets the <see cref="ILeastSquaresMethod">Least-Squares</see>
        ///   optimization algorithm used to perform the actual learning.
        /// </summary>
        /// 
        public ILeastSquaresMethod Algorithm
        {
            get { return solver; }
            set { solver = value; }
        }

        /// <summary>
        ///   Gets the number of variables (free parameters) in the non-linear model specified in <see cref="Function"/>.
        /// </summary>
        /// 
        /// <value>
        ///   The number of parameters of <see cref="Function"/>.
        /// </value>
        /// 
        public int NumberOfParameters
        {
            get { return numberOfParameters; }
            set { numberOfParameters = value; }
        }

        /// <summary>
        ///   Gets or sets the model function, mapping inputs to 
        ///   outputs given a suitable parameter vector.
        /// </summary>
        /// 
        public RegressionFunction Function
        {
            get { return function; }
            set { function = value; }
        }

        /// <summary>
        ///   Gets or sets a function that computes the gradient of the
        ///   <see cref="Function"/> in respect to the current parameters.
        /// </summary>
        /// 
        public RegressionGradientFunction Gradient
        {
            get { return gradient; }
            set { gradient = value; }
        }

        /// <summary>
        /// Gets or sets the vector of initial values to be used at the beginning
        /// of the optimization. Setting a suitable set of initial values can be
        /// important to achieve good convergence or avoid poor local minimas.
        /// </summary>
        /// 
        public double[] StartValues
        {
            get; set;
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="NonlinearLeastSquares" /> class.
        /// </summary>
        /// 
        public NonlinearLeastSquares()
        {

        }

        /// <summary>
        ///   Initializes a new instance of the <see cref="NonlinearLeastSquares"/> class.
        /// </summary>
        /// 
        /// <param name="regression">The regression model.</param>
        /// 
        public NonlinearLeastSquares(NonlinearRegression regression)
            : this(regression, new LevenbergMarquardt(regression.Coefficients.Length))
        {
        }

        /// <summary>
        ///   Initializes a new instance of the <see cref="NonlinearLeastSquares"/> class.
        /// </summary>
        /// 
        /// <param name="regression">The regression model.</param>
        /// <param name="algorithm">The <see cref="ILeastSquaresMethod">least squares</see>
        /// algorithm to be used to estimate the regression parameters. Default is to
        /// use a <see cref="LevenbergMarquardt">Levenberg-Marquardt</see> algorithm.</param>
        /// 
        public NonlinearLeastSquares(NonlinearRegression regression, ILeastSquaresMethod algorithm)
        {
            if (regression == null)
                throw new ArgumentNullException("regression");

            if (algorithm == null)
                throw new ArgumentNullException("algorithm");

            if (regression.Gradient == null)
                throw new ArgumentException("The regression must have a gradient function defined.", "regression");

            this.regression = regression;
            this.NumberOfParameters = regression.Coefficients.Length;

            this.solver = algorithm;
            this.solver.Solution = regression.Coefficients;
            this.solver.Function = new LeastSquaresFunction(regression.Function);
            this.solver.Gradient = new LeastSquaresGradientFunction(regression.Gradient);
        }

        /// <summary>
        /// Gets or sets a cancellation token that can be used to
        /// stop the learning algorithm while it is running.
        /// </summary>
        public CancellationToken Token
        {
            get { return token; }
            set { token = value; }
        }

        /// <summary>
        /// Learns a model that can map the given inputs to the given outputs.
        /// </summary>
        /// <param name="x">The model inputs.</param>
        /// <param name="y">The desired outputs associated with each <paramref name="x">inputs</paramref>.</param>
        /// <param name="weights">The weight of importance for each input-output pair (if supported by the learning algorithm).</param>
        /// <returns>
        /// A model that has learned how to produce <paramref name="y" /> given <paramref name="x" />.
        /// </returns>
        public NonlinearRegression Learn(double[][] x, double[] y, double[] weights = null)
        {
            if (weights != null)
                throw new ArgumentException(Accord.Properties.Resources.NotSupportedWeights, "weights");

            if (NumberOfParameters == 0)
            {
                if (regression == null)
                {
                    if (StartValues == null)
                        throw new InvalidOperationException("Please set the number of parameters, starting values, or the initial regression model.");
                    NumberOfParameters = StartValues.Length;
                }
            }

            if (regression == null)
            {
                this.regression = new NonlinearRegression(numberOfParameters, function, gradient);
                if (StartValues != null)
                    this.regression.Coefficients.SetTo(StartValues);
            }

            if (this.solver == null)
                this.solver = new LevenbergMarquardt(numberOfParameters);

            this.solver.NumberOfParameters = numberOfParameters;
            this.solver.Solution = regression.Coefficients;
            this.solver.Function = new LeastSquaresFunction(regression.Function);
            this.solver.Gradient = new LeastSquaresGradientFunction(regression.Gradient);
            this.solver.Token = Token;

            double error = solver.Minimize(x, y);

            if (Double.IsNaN(error) || Double.IsInfinity(error))
                throw new Exception();

            if (computeStandardErrors)
            {
                double[] errors = solver.StandardErrors;
                for (int i = 0; i < errors.Length; i++)
                    regression.StandardErrors[i] = solver.StandardErrors[i];
            }


            return regression;
        }
        /// <summary>
        /// Learns a model that can map the given inputs to the given outputs by using (gradient free) optimization algorithm
        /// </summary>
        /// <param name="x">The model inputs.</param>
        /// <param name="y">The desired outputs associated with each <paramref name="x">inputs</paramref>.</param>
        /// <typeparam name="T">The desired algorithm which shall be used</typeparam>
        /// <returns>
        /// A model that has learned how to produce <paramref name="y" /> given <paramref name="x" />.
        /// </returns>
        public NonlinearRegression Learn<T>(Double[][] x, double[] y) where T:BaseOptimizationMethod
        {
            NonlinearRegression problemFormulation = null;

            Func<double[][],double[],double[], double> error = ErrorFunction;
             var errorForInputAndOutput = error.Curry()(x)(y);

            var gradientFreeSolver = (BaseOptimizationMethod) Activator.CreateInstance(typeof(T), this.NumberOfParameters );
            gradientFreeSolver.Function = errorForInputAndOutput;
            gradientFreeSolver.NumberOfVariables = this.NumberOfParameters;

            bool success = gradientFreeSolver.Minimize();

            problemFormulation = new NonlinearRegression(gradientFreeSolver.NumberOfVariables,this.Function);

            for(int idx = 0; idx < gradientFreeSolver.Solution.Length;idx++)
            {
                problemFormulation.Coefficients.SetValue(gradientFreeSolver.Solution[idx],idx);
            }

            return problemFormulation;
        }
        /// <summary>
        /// Learns a model that can map the given inputs to the given outputs by using (gradient free) optimization algorithm
        /// </summary>
        /// <param name="x">The model inputs.</param>
        /// <param name="y">The desired outputs associated with each <paramref name="x">inputs</paramref>.</param>
        /// <param name="gradientFreeSolver">The optimization solver<paramref name="gradientFreeSolver">inputs</paramref>.</param>
        /// <returns>
        /// A model that has learned how to produce <paramref name="y" /> given <paramref name="x" />.
        /// </returns>
        public NonlinearRegression Learn(Double[][] x, double[] y, BaseOptimizationMethod gradientFreeSolver) 
        {
            NonlinearRegression problemFormulation = null;

            Func<double[][],double[],double[], double> error = ErrorFunction;
             var errorForInputAndOutput = error.Curry()(x)(y);

            gradientFreeSolver.Function = errorForInputAndOutput;
            gradientFreeSolver.NumberOfVariables = this.NumberOfParameters;

            bool success = gradientFreeSolver.Minimize();

            problemFormulation = new NonlinearRegression(gradientFreeSolver.NumberOfVariables,this.Function);

            for(int idx = 0; idx < gradientFreeSolver.Solution.Length;idx++)
            {
                problemFormulation.Coefficients.SetValue(gradientFreeSolver.Solution[idx],idx);
            }

            return problemFormulation;
        }

        /// <summary>
        /// Learns a model that can map the given inputs to the given outputs by using (gradient free) optimization algorithm
        /// </summary>
        /// <param name="input">The model inputs.</param>
        /// <param name="output">The desired outputs associated with each inputs.</param>
        /// <param name="parameter">The parameter for model .</param>
        /// <returns>
        /// The error between model output with parameter and true output.
        /// </returns>
        protected double ErrorFunction(double[][] input, double[] output, double[]parameter)
        {
            int points = input.Length;
            double error = 0;
            // invoke the function to compute the regression functions output
            // build the difference between computed output and true output, i.e. error
            // error with power of 2 and sum them up
            for (int idx = 0; idx < points; idx++)
            {
                error =  error + System.Math.Pow(Function.Invoke(parameter,input[idx]) - output[idx],2);
            }

            return error;
        }
    }
    // magic transformation to map a 4 parameter function to partial application
    // like in F#
    internal static class CurryExtension
    {
        internal static Func<A,Func<B,Func<C,D>>> Curry<A,B,C,D>(this Func<A,B,C,D> func) => a => b => c => func(a,b,c);
        internal static Func<A,Func<B,C>> Curry<A,B,C>(this Func<A,B,C> func) => a => b => func(a,b);
    }
}
