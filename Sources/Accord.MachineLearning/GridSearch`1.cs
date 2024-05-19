// Accord Machine Learning Library
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
    using Accord.MachineLearning.VectorMachines.Learning;
    using Accord.Statistics.Kernels;
    
    using System.Threading;
    using System.Threading.Tasks;

    /// <summary>
    ///   Delegate for grid search fitting functions.
    /// </summary>
    /// 
    /// <typeparam name="TModel">The type of the model to fit.</typeparam>
    /// 
    /// <param name="parameters">The collection of parameters to be used in the fitting process.</param>
    /// <param name="error">The error (or any other performance measure) returned by the model.</param>
    /// <returns>The model fitted to the data using the given parameters.</returns>
    /// 
    public delegate TModel GridSearchFittingFunction<TModel>(GridSearchParameterCollection parameters, out double error);

    /// <summary>
    ///   Contains results from the grid-search procedure.
    /// </summary>
    /// 
    /// <typeparam name="TModel">The type of the model to be tuned.</typeparam>
    /// 
    public class GridSearchResult<TModel> where TModel : class
    {

        private GridSearchParameterCollection[] parameters;
        private TModel[] models;
        private double[] errors;
        private int gridSize;
        private int bestIndex;

        /// <summary>
        ///   Gets all combination of parameters tried.
        /// </summary>
        /// 
        public GridSearchParameterCollection[] Parameters
        {
            get { return parameters; }
        }

        /// <summary>
        ///   Gets all models created during the search.
        /// </summary>
        /// 
        public TModel[] Models
        {
            get { return models; }
        }

        /// <summary>
        ///   Gets the error for each of the created models.
        /// </summary>
        /// 
        public double[] Errors
        {
            get { return errors; }
        }

        /// <summary>
        ///   Gets the index of the best found model
        ///   in the <see cref="Models"/> collection.
        /// </summary>
        /// 
        public int Index
        {
            get { return bestIndex; }
        }

        /// <summary>
        ///   Gets the best model found.
        /// </summary>
        /// 
        public TModel Model
        {
            get { return models[bestIndex]; }
        }

        /// <summary>
        ///   Gets the best parameter combination found.
        /// </summary>
        /// 
        public GridSearchParameterCollection Parameter
        {
            get { return parameters[bestIndex]; }
        }

        /// <summary>
        ///   Gets the minimum error found.
        /// </summary>
        /// 
        public double Error
        {
            get { return errors[bestIndex]; }
        }


        /// <summary>
        ///   Gets the size of the grid used in the grid-search.
        /// </summary>
        /// 
        public int Count
        {
            get { return gridSize; }
        }



        /// <summary>
        ///   Initializes a new instance of the <see cref="GridSearchResult&lt;TModel&gt;"/> class.
        /// </summary>
        /// 
        public GridSearchResult(int size)
        {
            gridSize = size;
            parameters = new GridSearchParameterCollection[size];
            models = new TModel[size];
            errors = new double[size];
        }

        /// <summary>
        ///   Initializes a new instance of the <see cref="GridSearchResult&lt;TModel&gt;"/> class.
        /// </summary>
        /// 
        public GridSearchResult(int size, GridSearchParameterCollection[] parameters,
            TModel[] models, double[] errors, int index)
        {
            this.gridSize = size;

            if (parameters.Length != size || models.Length != size || errors.Length != size)
                throw new DimensionMismatchException("size", "All array parameters must have the same length.");

            if (0 > index || index >= size)
                throw new ArgumentOutOfRangeException("index", "Index must be higher than 0 and less than size.");

            this.parameters = parameters;
            this.models = models;
            this.errors = errors;
            this.bestIndex = index;
        }

    }
}
