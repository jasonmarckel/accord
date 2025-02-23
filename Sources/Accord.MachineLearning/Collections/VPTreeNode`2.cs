// Accord Machine Learning Library
// The Accord.NET Framework
// http://accord-framework.net
//
// Copyright � C�sar Souza, 2009-2017
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

namespace Accord.Collections
{
    using System;
    using System.Linq;
    using System.Collections.Generic;
    
    
#if !MONO
    /// <summary>
    ///   Node of a <see cref="VPTree"/>.
    /// </summary>
    /// 
    /// <typeparam name="TPoint">The type for the position vector (e.g. double[]).</typeparam>
    /// <typeparam name="TData">The type for the value stored at the node.</typeparam>
    /// 
    [Serializable]
    public class VPTreeNode<TPoint, TData> : VPTreeNodeBase<TPoint, VPTreeNode<TPoint, TData>>
    {
        /// <summary>
        ///   Gets or sets a value associated with this node.
        /// </summary>
        /// 
        public TData Value { get; set; }

        /// <summary>
        /// Returns a <see cref="System.String" /> that represents this instance.
        /// </summary>
        /// <returns>
        /// A <see cref="System.String" /> that represents this instance.
        /// </returns>
        public override string ToString()
        {
            return String.Format("{0}: {1}", Value, base.ToString());
        }
    }
#endif
}