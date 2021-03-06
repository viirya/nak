/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package nak.core;



/**
 * Generate contexts for maxent decisions, assuming that the input
 * given to the getContext() method is a String containing contextual
 * predicates separated by spaces. 
 * e.g:
 * <p>
 * cp_1 cp_2 ... cp_n
 * </p>
 */
public class BasicContextGenerator implements ContextGenerator {

  private final String separator;
  private final boolean includeIntercept;
  private final String interceptFeature = "intercept=[**TRUE**]";

  public BasicContextGenerator () {
    this(" ");
  }
  
  public BasicContextGenerator (String sep) {
    this(sep, false);
  }

  public BasicContextGenerator (String sep, boolean includeIntercept) {
    separator = sep;
    this.includeIntercept = includeIntercept;
  }

  /**
   * Builds up the list of contextual predicates given a String.
   */
  public String[] getContext(Object o) {
    String s = (String) o;
    if (includeIntercept) {
      // Okay, so this is dumb, but it is just a quick hack to avoid 
      // Java collections pain.
      String interceptString = interceptFeature + separator + s;
      //System.out.println("&&&&&&&&&&&&&&& " + interceptString);
      return (String[]) interceptString.split(separator);
    } else {
      return (String[]) s.split(separator);
    }
  }
 
}

