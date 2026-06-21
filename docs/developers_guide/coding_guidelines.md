# Coding Guidelines

These guidelines are intended to keep new code explicit, testable, and loosely coupled. They are especially important when modifying shared infrastructure or code that is used across multiple modules.

## Avoid New Global-State Dependencies

Avoid introducing new dependencies on global state such as `GlobalV`, `GlobalC`, or `PARAM`.

These globals exist mainly as temporary compatibility mechanisms for legacy code. New code should prefer passing required data explicitly through function arguments or well-defined configuration objects. This makes dependencies visible at call sites and reduces accidental coupling between unrelated modules.

## Keep Control Flow Explicit

Do not use class member variables as hidden control flags for logic that is shared across functions or classes.

Such members behave like local global state: multiple methods can read or modify them without going through an explicit interface, which makes it easy to break the original logic. Prefer explicit parameters, return values, scoped helper objects, or clearly documented state-transition methods.

## Keep Header Dependencies Light

Be conservative when including header files from other header files.

Unnecessary `#include` directives in `.h` files create long dependency chains. A problem in one included header can then affect unrelated code that only depends on it indirectly. Prefer forward declarations when possible, and include full definitions in `.cpp` files unless the header genuinely needs them.

## Avoid Heavy `.hpp` Files

Avoid putting substantial implementation details in `.hpp` files unless there is a clear technical need, such as templates or performance-critical inline code.

Prefer splitting interfaces and implementations into `.h` and `.cpp` files. If a `.hpp` file is necessary, avoid including it from other header files unless required, because this can spread implementation dependencies widely through the codebase.

## Be Careful Around EXX Dependencies

Be especially careful when adding dependencies involving EXX-related code.

This area currently has a deep dependency chain and is expected to be refactored later to reduce coupling. Avoid expanding this dependency chain unless the change is necessary and well justified.

## Avoid New Default Arguments on Existing Functions

Do not add new default arguments to existing functions as a compatibility shortcut.

Even when intended to preserve old call sites, default arguments silently change how existing code compiles and can introduce behavior or dependencies that developers do not notice. Prefer making every option explicit at each call site so that each developer sees and considers the full function interface.

## Add Focused Tests for Important Behavior

For important functionality, add focused unit tests that verify correctness.

AI-generated tests are acceptable, but they should be short, targeted, and maintainable. Prefer testing the essential behavior and edge cases directly instead of creating broad or overly verbose tests.
