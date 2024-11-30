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

#pragma once



#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <stdexcept>
#include <sstream>

#include "command_line_options_export.h"


namespace mnist_deep_ann {



  // Struct CommandLine
  //   Represents the arguments passed in a command line
  struct COMMAND_LINE_OPTIONS_EXPORT CommandLine {
    int argc;
    char const * const * argv;
    CommandLine(int c, char const * const * v): argc(c), argv(v) {}

    bool isFinished() const { return (argc <= 0); }
    std::string extract();    // extract the next element, and advance the pointer
    std::string peek() const; // get the next element, but do NOT advance the pointer

  };




  // Interface CloInterface
  // Abstract interface so that templatized command line options 
  //   can be stored in a single vector.
  class CloInterface {
    public:

      std::string name;
      std::string help;
      const bool isMandatory;
      
      CloInterface(const std::string & v_name, const std::string & v_help, bool v_isMandatory):
        name(v_name),
        help(v_help),
        isMandatory(v_isMandatory)
      { }

      COMMAND_LINE_OPTIONS_EXPORT virtual bool parse(CommandLine & cmd) = 0;
      COMMAND_LINE_OPTIONS_EXPORT virtual bool isSet() const = 0;
      virtual ~CloInterface() { }
  };
  



  // Function parse
  //   Uses a sequence of definitions to parse a sequence of command line options.
  //   The parsed values are populated into the definitions.
  //   The "skipFirstArg" flag can be used to skip the name of the program itself.
  COMMAND_LINE_OPTIONS_EXPORT void parse(const std::vector<std::shared_ptr<CloInterface> > & defn, int argc, char const * const * argv, bool skipFirstArg = true);



  // Template class CommandLineOption
  //   Used for defining a command line option.
  template <typename TVal>
    class CommandLineOption : public CloInterface {
      public:
        std::optional<TVal> value;

        CommandLineOption(const std::string & name,
                          const std::string & help,
                          bool isMandatory = false,
                          const std::optional<TVal> & defaultValue = std::optional<TVal>()):
          CloInterface(name, help, isMandatory),
          value(defaultValue)
        { }

      public:
        bool parse(CommandLine & cl) override
        {
          std::string peek = cl.peek();
          if (peek != name)
            return false;
          cl.extract();
          if (cl.isFinished())
            throw std::runtime_error("Expected value after flag '" + name + "' in the command line.");
          std::stringstream ss;
          ss << cl.extract();
          TVal v;
          ss >> v;
          value = v;
          return true;
        }

        bool isSet() const override
        {
          return value.has_value();
        }

        static std::shared_ptr<CommandLineOption<TVal> > make(const std::string & name,
                                                  const std::string & help,
                                                  bool isMandatory = false,
                                                  std::optional<TVal> defaultValue = std::make_optional<TVal>())
        {
          return std::make_shared < CommandLineOption < TVal > >(name, help, isMandatory, defaultValue);
        }

    }; // end class CommandLineOption


} // end namespace mnist_deep_ann