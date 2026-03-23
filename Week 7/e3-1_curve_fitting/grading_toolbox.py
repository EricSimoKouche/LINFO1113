#!/usr/bin/env python3

# Copyright (c) 2024-2026, Sebastien Jodogne, ICTEAM UCLouvain, Belgium
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def __get_func_name(func):
    # Remove the decorators
    while func.__closure__ != None:
        func = func.__closure__[0].cell_contents
    
        if isinstance(func, str):  # This happens with "@unittest.skip"
            return None

    return func.__qualname__

__grade_feedbacks = {}
def grade_feedback(text):
    def decorator(func):
        def wrapper(self):
            global __grade_feedbacks
            name = __get_func_name(func)
            if name != None:
                __grade_feedbacks[name] = text
            func(self)
        return wrapper
    return decorator


__grades = {}
def grade(value):
    value = float(value)
    def decorator(func):
        def wrapper(self):
            global __grades
            name = __get_func_name(func)
            if name != None:
                __grades[name] = value
            func(self)
        return wrapper
    return decorator

def get_grade_feedbacks():
    return __grade_feedbacks

def get_grades():
    return __grades
