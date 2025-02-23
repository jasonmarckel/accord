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

namespace Accord.Statistics.Analysis
{
    using System;
    using Accord.Math;
    using Accord.Math.Comparers;
    using Accord.Math.Decompositions;
    using Accord.Statistics.Kernels;
    using Accord.MachineLearning;
    using System.Threading;
    using Models.Regression;


    /// <summary>
    ///   Kernel (Fisher) Discriminant Analysis.
    /// </summary>
    /// 
    /// <remarks>
    /// <para>
    ///   Kernel (Fisher) discriminant analysis (kernel FDA) is a non-linear generalization
    ///   of linear discriminant analysis (LDA) using techniques of kernel methods. Using a
    ///   kernel, the originally linear operations of LDA are done in a reproducing kernel
    ///   Hilbert space with a non-linear mapping.</para>
    /// <para>
    ///   The algorithm used is a multi-class generalization of the original algorithm by
    ///   Mika et al. in Fisher discriminant analysis with kernels (1999).</para>  
    ///   
    /// <para>
    ///   This class can also be bound to standard controls such as the 
    ///   <a href="http://msdn.microsoft.com/en-us/library/system.windows.forms.datagridview.aspx">DataGridView</a>
    ///   by setting their DataSource property to the analysis' <see cref="BaseDiscriminantAnalysis.Discriminants"/> property.</para>
    ///   
    /// <para>
    ///   References:
    ///   <list type="bullet">
    ///     <item><description>
    ///       Mika et al, Fisher discriminant analysis with kernels (1999). Available on
    ///       <a href="http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.35.9904">
    ///       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.35.9904 </a></description></item>
    ///  </list></para>  
    /// </remarks>
    /// 
    /// <example>
    /// <code source="Unit Tests\Accord.Tests.Statistics\Analysis\KernelDiscriminantAnalysisTest.cs" region="doc_learn" />
    /// </example>
    /// 
    [Serializable]
    public class KernelDiscriminantAnalysis : BaseDiscriminantAnalysis,
        ISupervisedLearning<KernelDiscriminantAnalysis.Pipeline, double[], int>
    {
        private IKernel kernel;
        private double regularization = 1e-4;
        private double[][] input;

        /// <summary>
        ///   Gets a classification pipeline that can be used to classify
        ///   new samples into one of the <see cref="BaseDiscriminantAnalysis.NumberOfClasses"/> 
        ///   learned in this discriminant analysis. This pipeline is
        ///   only available after a call to the <see cref="Learn"/> method.
        /// </summary>
        /// 
        public Pipeline Classifier { get; private set; }

        /// <summary>
        ///   Gets or sets the matrix of original values used to create
        ///   this analysis. Those values are required to build kernel 
        ///  (Gram) matrices when classifying new samples.
        /// </summary>
        /// 
        public double[][] Input
        {
            get { return this.input; }
            set { this.input = value; }
        }

        /// <summary>
        ///   Constructs a new Kernel Discriminant Analysis object.
        /// </summary>
        /// 
        public KernelDiscriminantAnalysis(IKernel kernel)
        {
            this.kernel = kernel;
            this.Threshold = 0;
        }

        /// <summary>
        ///   Constructs a new Kernel Discriminant Analysis object.
        /// </summary>
        /// 
        public KernelDiscriminantAnalysis()
            : this(new Linear())
        {

        }

        /// <summary>
        ///   Gets or sets the Kernel used in the analysis.
        /// </summary>
        /// 
        public IKernel Kernel
        {
            get { return kernel; }
            set { kernel = value; }
        }

        /// <summary>
        ///   Gets or sets the regularization parameter to
        ///   avoid non-singularities at the solution.
        /// </summary>
        /// 
        public double Regularization
        {
            get { return regularization; }
            set
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException("value", "Value must be positive.");
                regularization = value;
            }
        }

        private Pipeline CreateClassifier()
        {
            if (NumberOfOutputs == 0)
                return null;

            double[][] eig = DiscriminantVectors;

            return new Pipeline()
            {
                NumberOfInputs = NumberOfInputs,
                NumberOfOutputs = NumberOfClasses,
                NumberOfClasses = NumberOfClasses,

                First = new MultivariateKernelRegression()
                {
                    Weights = eig,
                    BasisVectors = input,
                    Kernel = Kernel,
                    NumberOfInputs = NumberOfInputs,
                    NumberOfOutputs = NumberOfOutputs,
                },
                Second = new MinimumMeanDistanceClassifier()
                {
                    Means = projectedMeans,
                    NumberOfInputs = NumberOfOutputs,
                    NumberOfOutputs = NumberOfClasses,
                    NumberOfClasses = NumberOfClasses,
                }
            };
        }

        /// <summary>
        /// Applies the transformation to an input, producing an associated output.
        /// </summary>
        /// <param name="input">The input data to which the transformation should be applied.</param>
        /// <param name="result">A location to store the output, avoiding unnecessary memory allocations.</param>
        /// <returns>
        /// The output generated by applying this transformation to the given input.
        /// </returns>
        public override double[][] Transform(double[][] input, double[][] result)
        {
            // TODO: Do without forming the kernel matrix
            double[][] K = kernel.ToJagged2(x: input, y: this.input);
            // return K.DotWithTransposed(DiscriminantVectors);

            for (int i = 0; i < input.Length; i++)
                for (int j = 0; j < result[i].Length; j++)
                    for (int k = 0; k < K[i].Length; k++)
                        result[i][j] += K[i][k] * DiscriminantVectors[j][k];
            return result;
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
        /// 
        public Pipeline Learn(double[][] x, int[] y, double[] weights = null)
        {
            if (weights != null)
                throw new ArgumentException(Accord.Properties.Resources.NotSupportedWeights, "weights");

            Init(x, y);

            // Create the Gram (Kernel) Matrix
            var K = kernel.ToJagged(x);

            // Compute entire data set measures
            base.Means = Measures.Mean(K, dimension: 0);
            base.StandardDeviations = Measures.StandardDeviation(K, Means);

            // Initialize the kernel analogous scatter matrices
            //int dimension = x.Columns();
            double[][] Sb = Jagged.Zeros(NumberOfSamples, NumberOfSamples);
            double[][] Sw = Jagged.Zeros(NumberOfSamples, NumberOfSamples);

            // For each class
            for (int c = 0; c < Classes.Count; c++)
            {
                var idx = Matrix.Find(y, y_i => y_i == c);

                // Get the Kernel matrix class subset
                double[][] Kc = K.Get(idx);
                int count = Kc.Rows();

                // Get the Kernel matrix class mean
                double[] mean = Measures.Mean(Kc, dimension: 0);

                // Construct the Kernel equivalent of the Within-Class Scatter matrix
                double[][] Swi = Measures.Scatter(Kc, dimension: 0, means: mean);
                Swi.Divide((double)count, result: Swi);
                Sw.Add(Swi, result: Sw); // Sw = Sw + Swi

                // Construct the Kernel equivalent of the Between-Class Scatter matrix
                double[] d = mean.Subtract(base.Means);
                double[][] Sbi = Jagged.Outer(d, d);
                Sbi.Multiply((double)NumberOfSamples, result: Sbi);

                Sb.Add(Sbi, result: Sb); // Sb = Sb + Sbi

                // Store additional information
                base.ClassScatter[c] = Swi;
                base.ClassCount[c] = count;
                base.ClassMeans[c] = mean;
                base.ClassStandardDeviations[c] = Measures.StandardDeviation(Kc, mean);
            }

            // Add regularization to avoid singularity
            Sw.AddToDiagonal(regularization, result: Sw);

            // Compute the generalized eigenvalue decomposition
            var gevd = new JaggedGeneralizedEigenvalueDecomposition(Sb, Sw, sort: true);

            if (gevd.IsSingular) // check validity of the results
            {
                throw new SingularMatrixException("One of the matrices is singular. Please retry " +
                    "the method with a higher regularization constant.");
            }

            // Get the eigenvalues and corresponding eigenvectors
            double[] evals = gevd.RealEigenvalues;
            double[][] eigs = gevd.Eigenvectors;

            // Eliminate unwanted components
            int nonzero = x.Columns();
            if (Threshold > 0)
                nonzero = Math.Min(gevd.Rank, GetNonzeroEigenvalues(evals, Threshold));
            if (NumberOfInputs != 0)
                nonzero = Math.Min(nonzero, NumberOfInputs);
            if (NumberOfOutputs != 0)
                nonzero = Math.Min(nonzero, NumberOfOutputs);

            eigs = eigs.Get(null, 0, nonzero);
            evals = evals.Get(0, nonzero);

            // Store information
            this.input = x;
            base.Eigenvalues = evals;
            base.DiscriminantVectors = eigs.Transpose();
            base.ScatterBetweenClass = Sb;
            base.ScatterWithinClass = Sw;
            base.NumberOfOutputs = evals.Length;

            // Compute feature space means for later classification
            for (int c = 0; c < Classes.Count; c++)
                ProjectionMeans[c] = ClassMeans[c].Dot(eigs);

            // Computes additional information about the analysis and creates the
            //  object-oriented structure to hold the discriminants found.
            CreateDiscriminants();

            this.Classifier = CreateClassifier();

            return Classifier;
        }

        /// <summary>
        ///   Gets the output of the discriminant function for a given class.
        /// </summary>
        /// 
        public override double DiscriminantFunction(double[] input, int classIndex)
        {
            return Classifier.Score(input, classIndex);
        }

        /// <summary>
        ///   Standard regression and classification pipeline for <see cref="LinearDiscriminantAnalysis"/>.
        /// </summary>
        /// 
        [Serializable]
        public sealed class Pipeline : MulticlassScoreClassifierBase<double[]>
        {
            /// <summary>
            /// Gets or sets the first step in the pipeline.
            /// </summary>
            /// 
            public MultivariateKernelRegression First { get; set; }

            /// <summary>
            /// Gets or sets the second step in the pipeline.
            /// </summary>
            /// 
            public MinimumMeanDistanceClassifier Second { get; set; }

            /// <summary>
            /// Computes a numerical score measuring the association between
            /// the given <paramref name="input" /> vector and each class.
            /// </summary>
            /// <param name="input">The input vector.</param>
            /// <param name="result">An array where the result will be stored,
            /// avoiding unnecessary memory allocations.</param>
            /// <returns></returns>
            public override double[][] Scores(double[][] input, double[][] result)
            {
                return Second.Scores(First.Transform(input), result);
            }

            /// <summary>
            /// Computes a numerical score measuring the association between
            /// the given <paramref name="input" /> vector and a given
            /// <paramref name="classIndex" />.
            /// </summary>
            /// <param name="input">The input vector.</param>
            /// <param name="classIndex">The index of the class whose score will be computed.</param>
            /// <returns>System.Double.</returns>
            public override double Score(double[] input, int classIndex)
            {
                return Second.Score(First.Transform(input), classIndex);
            }
        }
    }
}
