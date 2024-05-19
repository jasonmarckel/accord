// Accord Statistics Library
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

namespace Accord.Statistics.Models.Regression.Linear
{
    using System;
    using Accord.Math;
    using Accord.Math.Decompositions;
    using Accord.MachineLearning;
    using Accord.Statistics.Models.Regression.Fitting;
    using Accord.Math.Optimization.Losses;
    using Accord.Statistics.Testing;


    /// <summary>
    ///   Multivariate Linear Regression.
    /// </summary>
    /// 
    /// <remarks>
    ///   Multivariate Linear Regression is a generalization of
    ///   Multiple Linear Regression to allow for multiple outputs.
    /// </remarks>
    /// 
    /// <example>
    /// <code source="Unit Tests\Accord.Tests.Statistics\Models\Regression\MultivariateLinearRegressionTest.cs" region="doc_learn" />
    /// </example>
    /// 
    [Serializable]
#pragma warning disable 612, 618
    public class MultivariateLinearRegression : MultipleTransformBase<double[], double>
#pragma warning restore 612, 618
    {
        private double[][] weights;

        private double[,] coefficients; // obsolete
        private double[] intercepts;

        /// <summary>
        ///   Initializes a new instance of the <see cref="MultivariateLinearRegression"/> class.
        /// </summary>
        /// 
        public MultivariateLinearRegression()
        {
        }

        /// <summary>
        ///   Gets the linear weights matrix.
        /// </summary>
        /// 
        public double[][] Weights
        {
            get { return weights; }
            set
            {
                weights = value;
                NumberOfInputs = value.Rows();
                NumberOfOutputs = value.Columns();
                coefficients = value.ToMatrix();
            }
        }

        /// <summary>
        ///   Gets the intercept vector (bias).
        /// </summary>
        /// 
        public double[] Intercepts
        {
            get { return intercepts; }
            set { intercepts = value; }
        }

        /// <summary>
        ///   Gets the number of parameters in the model (returns 
        ///   NumberOfInputs * NumberOfOutputs + NumberOfInputs).
        /// </summary>
        /// 
        public int NumberOfParameters
        {
            get { return NumberOfInputs * NumberOfOutputs + NumberOfOutputs; }
        }

        /// <summary>
        ///   Gets the coefficient of determination, as known as R² (r-squared).
        /// </summary>
        /// 
        /// <remarks>
        ///   <para>
        ///    The coefficient of determination is used in the context of statistical models
        ///    whose main purpose is the prediction of future outcomes on the basis of other
        ///    related information. It is the proportion of variability in a data set that
        ///    is accounted for by the statistical model. It provides a measure of how well
        ///    future outcomes are likely to be predicted by the model.</para>
        ///   <para>
        ///    The R² coefficient of determination is a statistical measure of how well the
        ///    regression line approximates the real data points. An R² of 1.0 indicates
        ///    that the regression line perfectly fits the data.</para> 
        ///   <para>
        ///    This method uses the <see cref="RSquaredLoss"/> class to compute the R²
        ///    coefficient. Please see the documentation for <see cref="RSquaredLoss"/>
        ///    for more details, including usage examples.</para>
        /// </remarks>
        /// 
        /// <returns>The R² (r-squared) coefficient for the given data.</returns>
        /// 
        /// <seealso cref="RSquaredLoss"/>
        /// 
        public double[] CoefficientOfDetermination(double[][] inputs, double[][] outputs, double[] weights = null)
        {
            return CoefficientOfDetermination(inputs, outputs, false, weights);
        }

        /// <summary>
        ///   Gets the coefficient of determination, as known as R² (r-squared).
        /// </summary>
        /// 
        /// <remarks>
        ///   <para>
        ///    The coefficient of determination is used in the context of statistical models
        ///    whose main purpose is the prediction of future outcomes on the basis of other
        ///    related information. It is the proportion of variability in a data set that
        ///    is accounted for by the statistical model. It provides a measure of how well
        ///    future outcomes are likely to be predicted by the model.</para>
        ///   <para>
        ///    The R² coefficient of determination is a statistical measure of how well the
        ///    regression line approximates the real data points. An R² of 1.0 indicates
        ///    that the regression line perfectly fits the data.</para> 
        ///   <para>
        ///    This method uses the <see cref="RSquaredLoss"/> class to compute the R²
        ///    coefficient. Please see the documentation for <see cref="RSquaredLoss"/>
        ///    for more details, including usage examples.</para>
        /// </remarks>
        /// 
        /// <returns>The R² (r-squared) coefficient for the given data.</returns>
        /// 
        /// <seealso cref="RSquaredLoss"/>
        /// 
        public double[] CoefficientOfDetermination(double[][] inputs, double[][] outputs, bool adjust, double[] weights = null)
        {
            var rsquared = new RSquaredLoss(NumberOfInputs, outputs);

            rsquared.Adjust = adjust;

            if (weights != null)
                rsquared.Weights = weights;

            return rsquared.Loss(Transform(inputs));
        }

        /// <summary>
        ///   Creates a new linear regression directly from data points.
        /// </summary>
        /// 
        /// <param name="x">The input vectors <c>x</c>.</param>
        /// <param name="y">The output vectors <c>y</c>.</param>
        /// 
        /// <returns>A linear regression f(x) that most approximates y.</returns>
        /// 
        public static MultivariateLinearRegression FromData(double[][] x, double[][] y)
        {
            return new OrdinaryLeastSquares().Learn(x, y);
        }

        /// <summary>
        ///   Creates the inverse regression, a regression that can recover
        ///   the input data given the outputs of this current regression.
        /// </summary>
        /// 
        public MultivariateLinearRegression Inverse()
        {
            double[][] inv = Weights.PseudoInverse();
            return new MultivariateLinearRegression()
            {
                Weights = inv,
                Intercepts = intercepts != null ? intercepts.Dot(inv).Multiply(-1) : null
            };
        }

        /// <summary>
        /// Applies the transformation to an input, producing an associated output.
        /// </summary>
        /// <param name="input">The input data to which the transformation should be applied.</param>
        /// <param name="result"></param>
        /// <returns>
        /// The output generated by applying this transformation to the given input.
        /// </returns>
        public override double[][] Transform(double[][] input, double[][] result)
        {
            input.Dot(Weights, result: result);
            if (intercepts != null)
                result.Add(intercepts, dimension: (VectorType)0, result: result);
            return result;
        }

        /// <summary>
        /// Gets the overall regression standard error.
        /// </summary>
        /// 
        /// <param name="inputs">The inputs used to train the model.</param>
        /// <param name="outputs">The outputs used to train the model.</param>
        /// 
        public double[] GetStandardError(double[][] inputs, double[][] outputs)
        {
            // Calculate actual outputs (results)
            double[][] results = Transform(inputs);

            double[] SSe = new double[NumberOfOutputs];
            for (int i = 0; i < inputs.Length; i++)
            {
                for (int j = 0; j < SSe.Length; j++)
                {
                    double d = outputs[i][j] - results[i][j];
                    SSe[j] += d * d;
                }
            }

            double DFe = GetDegreesOfFreedom(inputs.Length);
            return Elementwise.Sqrt(SSe.Divide(DFe));
        }

        /// <summary>
        /// Gets the degrees of freedom when fitting the regression.
        /// </summary>
        /// 
        public double GetDegreesOfFreedom(int numberOfSamples)
        {
            return numberOfSamples - NumberOfParameters;
        }

        /// <summary>
        /// Gets the standard error for each coefficient.
        /// </summary>
        /// 
        /// <param name="mse">The overall regression standard error (can be computed from <see cref="GetStandardError(double[][], double[][])"/>.</param>
        /// <param name="informationMatrix">The information matrix obtained when training the model (see <see cref="OrdinaryLeastSquares.GetInformationMatrix()"/>).</param>
        /// 
        public double[][] GetStandardErrors(double[] mse, double[][] informationMatrix)
        {
            double[][] se = Jagged.Zeros(NumberOfOutputs, informationMatrix.Length);
            for (int j = 0; j < se.Length; j++)
                for (int i = 0; i < informationMatrix.Length; i++)
                    se[j][i] = mse[j] * Math.Sqrt(informationMatrix[i][i]);
            return se;
        }

        /// <summary>
        /// Gets the standard error of the fit for a particular input vector.
        /// </summary>
        /// 
        /// <param name="input">The input vector where the standard error of the fit should be computed.</param>
        /// <param name="mse">The overall regression standard error (can be computed from <see cref="GetStandardError(double[][], double[][])"/>.</param>        
        /// <param name="informationMatrix">The information matrix obtained when training the model (see <see cref="OrdinaryLeastSquares.GetInformationMatrix()"/>).</param>
        /// 
        /// <returns>The standard error of the fit at the given input point.</returns>
        /// 
        public double[] GetStandardError(double[] input, double[] mse, double[][] informationMatrix)
        {
            double rim = predictionVariance(input, informationMatrix);
            return mse.Multiply(Math.Sqrt(rim));
        }

        /// <summary>
        /// Gets the standard error of the prediction for a particular input vector.
        /// </summary>
        /// 
        /// <param name="input">The input vector where the standard error of the prediction should be computed.</param>
        /// <param name="mse">The overall regression standard error (can be computed from <see cref="GetStandardError(double[][], double[][])"/>.</param>
        /// <param name="informationMatrix">The information matrix obtained when training the model (see <see cref="OrdinaryLeastSquares.GetInformationMatrix()"/>).</param>
        /// 
        /// <returns>The standard error of the prediction given for the input point.</returns>
        /// 
        public double[] GetPredictionStandardError(double[] input, double[] mse, double[][] informationMatrix)
        {
            double rim = predictionVariance(input, informationMatrix);
            return mse.Multiply(Math.Sqrt(1 + rim));
        }

        /// <summary>
        /// Gets the confidence interval for an input point.
        /// </summary>
        /// 
        /// <param name="input">The input vector.</param>
        /// <param name="mse">The overall regression standard error (can be computed from <see cref="GetStandardError(double[][], double[][])"/>.</param>
        /// <param name="numberOfSamples">The number of training samples used to fit the model.</param>
        /// <param name="informationMatrix">The information matrix obtained when training the model (see <see cref="OrdinaryLeastSquares.GetInformationMatrix()"/>).</param>
        /// <param name="percent">The prediction interval confidence (default is 95%).</param>
        /// 
        public DoubleRange[] GetConfidenceInterval(double[] input, double[] mse, int numberOfSamples, double[][] informationMatrix, double percent = 0.95)
        {
            double[] se = GetStandardError(input, mse, informationMatrix);
            return createInterval(input, numberOfSamples, percent, se);
        }

        /// <summary>
        /// Gets the prediction interval for an input point.
        /// </summary>
        /// 
        /// <param name="input">The input vector.</param>
        /// <param name="mse">The overall regression standard error (can be computed from <see cref="GetStandardError(double[][], double[][])"/>.</param>
        /// <param name="numberOfSamples">The number of training samples used to fit the model.</param>
        /// <param name="informationMatrix">The information matrix obtained when training the model (see <see cref="OrdinaryLeastSquares.GetInformationMatrix()"/>).</param>
        /// <param name="percent">The prediction interval confidence (default is 95%).</param>
        /// 
        public DoubleRange[] GetPredictionInterval(double[] input, double[] mse, int numberOfSamples, double[][] informationMatrix, double percent = 0.95)
        {
            double[] se = GetPredictionStandardError(input, mse, informationMatrix);
            return createInterval(input, numberOfSamples, percent, se);
        }

        private static double predictionVariance(double[] input, double[][] im)
        {
            if (input.Length < im.Length)
                input = input.Concatenate(1);
            return input.Dot(im).Dot(input);
        }

        private DoubleRange[] createInterval(double[] input, int numberOfSamples, double percent, double[] se)
        {
            double[] y = Transform(input);
            double df = GetDegreesOfFreedom(numberOfSamples);
            return se.Apply((x, i) =>
                new TTest(estimatedValue: y[i], standardError: x, degreesOfFreedom: df)
                    .GetConfidenceInterval(percent));
        }
    }
}
