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

namespace Accord.MachineLearning.DecisionTrees
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    
    using System.Linq;

    /// <summary>
    ///   Collection of decision nodes. A decision branch specifies the index of
    ///   an attribute whose current value should be compared against its children
    ///   nodes. The type of the comparison is specified in each child node.
    /// </summary>
    /// 
    [Serializable]
    public class DecisionBranchNodeCollection : Collection<DecisionNode>
    {

        [NonSerialized]
        private DecisionNode owner;

        /// <summary>
        ///   Gets or sets the index of the attribute to be
        ///   used in this stage of the decision process.
        /// </summary>
        /// 
        public int AttributeIndex { get; set; }

        /// <summary>
        ///   Gets the attribute that is being used in
        ///   this stage of the decision process, given
        ///   by the current <see cref="AttributeIndex"/>
        /// </summary>
        /// 
        public DecisionVariable Attribute
        {
            get
            {
                // TODO: Remove the obsolete attribute and make owner mandatory.
                if (owner == null)
                    return null;

                return owner.Owner.Attributes[AttributeIndex];
            }
        }

        /// <summary>
        ///   Gets or sets the decision node that contains this collection.
        /// </summary>
        /// 
        public DecisionNode Owner
        {
            get { return owner; }
            set { owner = value; }
        }

        /// <summary>
        ///   Initializes a new instance of the <see cref="DecisionBranchNodeCollection"/> class.
        /// </summary>
        /// 
        /// <param name="owner">The <see cref="DecisionNode"/> to whom
        ///   this <see cref="DecisionBranchNodeCollection"/> belongs.</param>
        /// 
        public DecisionBranchNodeCollection(DecisionNode owner)
        {
            this.owner = owner;
        }

        /// <summary>
        ///   Adds the elements of the specified collection to the end of the collection.
        /// </summary>
        /// 
        /// <param name="children">The child nodes to be added.</param>
        /// 
        public void AddRange(IEnumerable<DecisionNode> children)
        {
            foreach (DecisionNode node in children)
                Add(node);
        }

        /// <summary>
        /// Returns a <see cref="System.String" /> that represents this instance.
        /// </summary>
        /// 
        /// <returns>A <see cref="System.String" /> that represents this instance.</returns>
        /// 
        public override string ToString()
        {
            string values = String.Join(", ", this.Select(x => x.ToString()).ToArray());
            return String.Format("{0} ({1}) => {2}", this.Attribute, this.AttributeIndex, values);
        }
    }
}
