﻿// Accord Machine Learning Library
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

namespace Accord.MachineLearning
{
    using System;
    using Accord.Math;
    using Accord.Math.Distances;
    
    using System.Threading.Tasks;
    using System.Runtime.CompilerServices;
    using System.Threading;
    using System.Runtime.Serialization;

    /// <summary>
    ///   K-Nearest Neighbor (k-NN) algorithm.
    /// </summary>
    /// 
    /// <typeparam name="TInput">The type of the input data.</typeparam>
    /// 
    /// <remarks>
    /// <para> The k-nearest neighbor algorithm (k-NN) is a method for classifying objects
    ///   based on closest training examples in the feature space. It is amongst the simplest
    ///   of all machine learning algorithms: an object is classified by a majority vote of
    ///   its neighbors, with the object being assigned to the class most common amongst its 
    ///   k nearest neighbors (k is a positive integer, typically small).</para>
    ///   
    /// <para>If k = 1, then the object is simply assigned to the class of its nearest neighbor.</para>
    /// 
    /// <note type="note">
    ///   When learning a model with instance weights, the weights will not be used when
    ///   finding the <c>k</c> nearest neighbors of a query point. Instead, it will be used
    ///   to weight the similarity between the query point and each of its <c>k</c> nearest
    ///   neighbors when deciding for the queried point's class.
    /// </note>
    /// 
    /// <para>
    ///   References:
    ///   <list type="bullet">
    ///     <item><description>
    ///       Wikipedia contributors. "K-nearest neighbor algorithm." Wikipedia, The
    ///       Free Encyclopedia. Wikipedia, The Free Encyclopedia, 10 Oct. 2012. Web.
    ///       9 Nov. 2012. http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm </description></item>
    ///   </list>
    /// </para>
    /// </remarks>
    /// 
    /// <example>
    /// <para>
    ///   The first example shows how to create and use a k-Nearest Neighbor algorithm to classify
    ///   a set of numeric vectors in a multi-class decision problem involving 3 classes. It also shows
    ///   how to compute class decisions for a new sample and how to measure the performance of a classifier.</para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\KNearestNeighbors\KNearestNeighborsTest.cs" region="doc_learn" />
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\KNearestNeighbors\KNearestNeighborsTest.cs" region="doc_serialization" />
    /// 
    /// <para>
    ///   The second example show how to use a different distance metric when computing k-NN:</para>
    ///   <code source = "Unit Tests\Accord.Tests.MachineLearning\KNearestNeighbors\KNearestNeighborsTest.cs" region="doc_learn_distance" />
    /// 
    /// <para>
    ///   The k-Nearest neighbor algorithm implementation in the framework can also be used with any instance 
    ///   data type. For such cases, the framework offers a generic version of the classifier. The third example
    ///   shows how to use the generic kNN classifier to perform the direct classification of actual text samples:</para>
    /// <code source = "Unit Tests\Accord.Tests.MachineLearning\KNearestNeighbors\KNearestNeighborsTest.cs" region="doc_learn_text" />
    /// </example>
    /// 
    /// <seealso cref="KNearestNeighbors"/>
    /// 
    [Serializable]
    public class KNearestNeighbors<TInput> :
        BaseKNearestNeighbors<KNearestNeighbors<TInput>, TInput, IDistance<TInput>>,
        IParallel
    {
        // TODO: After removing obsolete methods, mark class as sealed

        [NonSerialized]
        private ThreadLocal<double[]> distanceCache = new ThreadLocal<double[]>();

        [NonSerialized]
        private ParallelOptions parallelOptions = new ParallelOptions();

        /// <summary>
        ///   Gets or sets the parallelization options for this algorithm.
        /// </summary>
        /// 
        public ParallelOptions ParallelOptions
        {
            get { return parallelOptions; }
            set { parallelOptions = value; }
        }


        /// <summary>
        ///   Creates a new <see cref="KNearestNeighbors"/>.
        /// </summary>
        /// 
        public KNearestNeighbors()
        {
        }

        /// <summary>
        ///   Creates a new <see cref="KNearestNeighbors"/>.
        /// </summary>
        /// 
        public KNearestNeighbors(int k, IDistance<TInput> distance)
        {
            this.K = k;
            this.Distance = distance;
        }


        /// <summary>
        /// Computes a numerical score measuring the association between
        /// the given <paramref name="input" /> vector and each class.
        /// </summary>
        /// <param name="input">The input vector.</param>
        /// <param name="result">An array where the result will be stored,
        /// avoiding unnecessary memory allocations.</param>
        /// <returns>System.Double[].</returns>
        public override double[] Scores(TInput input, double[] result)
        {
            double[] distances;
            int[] idx = getNearestIndices(input, out distances);

            // Compute the scores for these points
            for (int i = 0; i < idx.Length; i++)
            {
                int j = idx[i];

                int label = Outputs[j];
                double weight = Weights[j];
                double d = distances[i]; // distances vector is already in order because
                                         // the bottom operation below was done in-place

                // Convert to similarity measure
                result[label] += 1.0 / (1.0 + d) * weight;
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int[] getNearestIndices(TInput input, out double[] distances)
        {
            double[] d = this.distanceCache.Value;

            if (this.parallelOptions.MaxDegreeOfParallelism == 1)
            {
                // Compute all distances
                for (int i = 0; i < Inputs.Length; i++)
                    d[i] = Distance.Distance(input, Inputs[i]);
            }
            else
            {
                // Compute all distances
                Parallel.For(0, Inputs.Length, this.parallelOptions, i =>
                {
                    d[i] = Distance.Distance(input, Inputs[i]);
                });
            }

            distances = d;

            // Get the K closest points
            return d.Bottom(K, inPlace: true);
        }

        /// <summary>
        ///   Gets the top <see cref="BaseKNearestNeighbors{TModel, TInput, TDistance}.K"/> points
        ///   that are the closest to a given <paramref name="input"> reference point</paramref>.
        /// </summary>
        /// 
        /// <param name="input">The query point whose neighbors will be found.</param>
        /// <param name="labels">The label for each neighboring point.</param>
        /// 
        /// <returns>
        ///   An array containing the top <see cref="BaseKNearestNeighbors{TModel, TInput, TDistance}.K"/> points that are 
        ///   at the closest possible distance to <paramref name="input"/>.
        /// </returns>
        /// 
        public override TInput[] GetNearestNeighbors(TInput input, out int[] labels)
        {
            double[] distances;
            int[] idx = getNearestIndices(input, out distances);

            labels = this.Outputs.Get(idx);
            return this.Inputs.Get(idx);
        }



        /// <summary>
        ///   Learns a model that can map the given inputs to the given outputs.
        /// </summary>
        /// 
        /// <param name="x">The model inputs.</param>
        /// <param name="y">The desired outputs associated with each <paramref name="x">inputs</paramref>.</param>
        /// <param name="weights">The weight of importance for each input-output pair (if supported by the learning algorithm).</param>
        /// 
        /// <returns>A model that has learned how to produce <paramref name="y"/> given <paramref name="x"/>.</returns>
        /// 
        public override KNearestNeighbors<TInput> Learn(TInput[] x, int[] y, double[] weights = null)
        {
            CheckArgs(K, x, y, Distance, weights);

            this.Inputs = x;
            this.Outputs = y;
            this.Weights = weights ?? Vector.Ones(x.Length);

            this.NumberOfInputs = GetNumberOfInputs(x);
            this.NumberOfOutputs = y.DistinctCount();
            this.NumberOfClasses = this.NumberOfOutputs;
            this.distanceCache = new ThreadLocal<double[]>(() => new double[Inputs.Length]);

            return this;
        }



        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.distanceCache = new ThreadLocal<double[]>(() => new double[Inputs.Length]);
            this.parallelOptions = new ParallelOptions();
            if (this.Weights == null && this.Inputs != null)
                this.Weights = Vector.Ones(this.Inputs.Length);
        }

    }
}
