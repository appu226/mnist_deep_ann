/*

Copyright 2020 Parakram Majumdar

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "clo.hpp"

#include <iostream>


// helper functions internal to this file
namespace
{


  // print help and exit
  void printHelp(const std::vector<std::shared_ptr<mnist_deep_ann::CloInterface> > & defnVec)
  {
    std::cout << "--help/-h:\tPrint this help and exit.\n";
    for (auto defn: defnVec)
    {
      std::cout << defn->name << ":\t" << defn->help;
      if (defn->isMandatory)
        std::cout << "\t[Mandatory]";
      std::cout << "\n";
    }
    std::cout << std::endl;
  }


} // end anonymous namespace









namespace mnist_deep_ann
{


  // **** Function definitions for struct CommandLine ****
  std::string CommandLine::extract() {
    if (argc <= 0)
      throw std::runtime_error("Attempted to read command line paramter when there are none remaining.");
    --argc;
    return *(argv++);
  }

  std::string CommandLine::peek() const {
    if (argc <= 0)
      throw std::runtime_error("Attempted to read command line paramter when there are none remaining.");
    return *argv;
  }




  // ***** Definition for function parse *****
  void parse(const std::vector<std::shared_ptr<CloInterface> > & defn, int argc, char const * const * argv, bool skipFirstArg)
  {
    // Skip the first argument (i.e., the name of the program itself).
    if(skipFirstArg)
    {
      --argc;
      ++argv;
    }

    // loop through all the tokens
    CommandLine cl(argc, argv);
    while(!cl.isFinished())
    {

      // print help and exit if needed
      std::string next = cl.peek();
      if (next == "--help" || next == "-h")
      {
        printHelp(defn);
        exit(0);
      }

      // loop through all possible option definitions to see if any matches
      bool isParsed = false;
      for (auto itDefn = defn.begin(); itDefn != defn.end() && !isParsed; ++itDefn)
        isParsed = (**itDefn).parse(cl);
      
      // if no options matched, then throw error
      if (!isParsed)
      {
        std::cerr << "Could not parse command line option '" << cl.peek() << "', use --help for guidance." << std::endl;
        printHelp(defn);
        exit(1);
      }
    } // end while loop


    for (auto d: defn)
    {
      if (d->isMandatory && !d->isSet())
      {
        std::cerr << "Missing mandatory parameter '" << d->name << "', use --help for guidance." << std::endl;
        printHelp(defn);
        exit(1);
      }
    }

  } // end function parse


} // end namespace mnist_deep_ann