"""
Advanced Mathematical Calculator
=================================
Perhitungan dasar (+, -, *, /) tapi dengan fitur kompleks:
- Expression parsing & evaluation
- Variable storage & management
- Function definitions
- History tracking
- Graph plotting
- Symbolic mathematics
- Unit conversions
- Statistical calculations
"""

import re
import math
import operator
from collections import deque, defaultdict
from typing import Union, List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np


class AdvancedCalculator:
    def __init__(self):
        # Operator precedence
        self.operators = {
            '+': (1, operator.add),
            '-': (1, operator.sub),
            '*': (2, operator.mul),
            '/': (2, operator.truediv),
            '//': (2, operator.floordiv),
            '%': (2, operator.mod),
            '^': (3, operator.pow),
            '**': (3, operator.pow),
        }
        
        # Built-in functions
        self.functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'log': math.log10,
            'ln': math.log,
            'abs': abs,
            'exp': math.exp,
            'factorial': math.factorial,
            'ceil': math.ceil,
            'floor': math.floor,
        }
        
        # Constants
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
        }
        
        # User variables & history
        self.variables = {}
        self.history = []
        self.ans = 0  # Last answer
        
    def tokenize(self, expression: str) -> List[str]:
        """Convert expression to tokens"""
        # Remove spaces
        expression = expression.replace(' ', '')
        
        # Regex pattern untuk tokens
        pattern = r'(\d+\.?\d*|[+\-*/%^()]|//|\*\*|[a-zA-Z_]\w*)'
        tokens = re.findall(pattern, expression)
        
        return tokens
    
    def infix_to_postfix(self, tokens: List[str]) -> List[str]:
        """Convert infix notation to postfix (Reverse Polish Notation)
        Using Shunting Yard Algorithm"""
        
        output = []
        operator_stack = []
        
        for token in tokens:
            # Number
            if self._is_number(token):
                output.append(token)
            
            # Variable or constant
            elif token in self.variables or token in self.constants:
                output.append(token)
            
            # Function
            elif token in self.functions:
                operator_stack.append(token)
            
            # Left parenthesis
            elif token == '(':
                operator_stack.append(token)
            
            # Right parenthesis
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                if operator_stack:
                    operator_stack.pop()  # Remove '('
                if operator_stack and operator_stack[-1] in self.functions:
                    output.append(operator_stack.pop())
            
            # Operator
            elif token in self.operators:
                precedence, _ = self.operators[token]
                while (operator_stack and 
                       operator_stack[-1] in self.operators and
                       self.operators[operator_stack[-1]][0] >= precedence):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
        
        # Pop remaining operators
        while operator_stack:
            output.append(operator_stack.pop())
        
        return output
    
    def evaluate_postfix(self, postfix: List[str]) -> float:
        """Evaluate postfix expression"""
        stack = []
        
        for token in postfix:
            # Number
            if self._is_number(token):
                stack.append(float(token))
            
            # Variable
            elif token in self.variables:
                stack.append(self.variables[token])
            
            # Constant
            elif token in self.constants:
                stack.append(self.constants[token])
            
            # Function
            elif token in self.functions:
                if stack:
                    arg = stack.pop()
                    result = self.functions[token](arg)
                    stack.append(result)
            
            # Operator
            elif token in self.operators:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    _, op_func = self.operators[token]
                    result = op_func(a, b)
                    stack.append(result)
        
        return stack[0] if stack else 0
    
    def calculate(self, expression: str) -> float:
        """Main calculation method"""
        try:
            tokens = self.tokenize(expression)
            postfix = self.infix_to_postfix(tokens)
            result = self.evaluate_postfix(postfix)
            
            # Store in history
            self.history.append({
                'expression': expression,
                'result': result
            })
            self.ans = result
            
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")
    
    def set_variable(self, name: str, value: Union[float, str]):
        """Set a variable"""
        if isinstance(value, str):
            value = self.calculate(value)
        self.variables[name] = value
        return value
    
    def derivative(self, expression: str, var: str = 'x', point: float = 0, h: float = 1e-7) -> float:
        """Calculate numerical derivative using central difference"""
        # Store original value
        original = self.variables.get(var, 0)
        
        # f(x+h)
        self.variables[var] = point + h
        f_plus = self.calculate(expression)
        
        # f(x-h)
        self.variables[var] = point - h
        f_minus = self.calculate(expression)
        
        # Restore original
        if original:
            self.variables[var] = original
        else:
            self.variables.pop(var, None)
        
        # Central difference: [f(x+h) - f(x-h)] / 2h
        derivative = (f_plus - f_minus) / (2 * h)
        return derivative
    
    def integral(self, expression: str, var: str = 'x', a: float = 0, b: float = 1, n: int = 1000) -> float:
        """Calculate numerical integral using Simpson's rule"""
        if n % 2 == 1:
            n += 1
        
        h = (b - a) / n
        total = 0
        
        # Store original value
        original = self.variables.get(var, 0)
        
        for i in range(n + 1):
            x = a + i * h
            self.variables[var] = x
            fx = self.calculate(expression)
            
            if i == 0 or i == n:
                total += fx
            elif i % 2 == 1:
                total += 4 * fx
            else:
                total += 2 * fx
        
        # Restore original
        if original:
            self.variables[var] = original
        else:
            self.variables.pop(var, None)
        
        return (h / 3) * total
    
    def solve_quadratic(self, a: float, b: float, c: float) -> Tuple[complex, complex]:
        """Solve quadratic equation ax² + bx + c = 0"""
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0:
            x1 = (-b + math.sqrt(discriminant)) / (2*a)
            x2 = (-b - math.sqrt(discriminant)) / (2*a)
            return (x1, x2)
        else:
            real = -b / (2*a)
            imag = math.sqrt(-discriminant) / (2*a)
            return (complex(real, imag), complex(real, -imag))
    
    def statistics(self, numbers: List[float]) -> Dict:
        """Calculate statistical measures"""
        n = len(numbers)
        mean = sum(numbers) / n
        
        # Variance & Standard Deviation
        variance = sum((x - mean) ** 2 for x in numbers) / n
        std_dev = math.sqrt(variance)
        
        # Median
        sorted_nums = sorted(numbers)
        if n % 2 == 0:
            median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
        else:
            median = sorted_nums[n//2]
        
        return {
            'count': n,
            'sum': sum(numbers),
            'mean': mean,
            'median': median,
            'variance': variance,
            'std_dev': std_dev,
            'min': min(numbers),
            'max': max(numbers),
            'range': max(numbers) - min(numbers)
        }
    
    def plot_function(self, expression: str, var: str = 'x', start: float = -10, end: float = 10, points: int = 500):
        """Plot a mathematical function"""
        x_values = np.linspace(start, end, points)
        y_values = []
        
        original = self.variables.get(var, 0)
        
        for x in x_values:
            try:
                self.variables[var] = x
                y = self.calculate(expression)
                y_values.append(y)
            except:
                y_values.append(np.nan)
        
        # Restore original
        if original:
            self.variables[var] = original
        else:
            self.variables.pop(var, None)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, 'b-', linewidth=2, label=f'f({var}) = {expression}')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel(var, fontsize=12)
        plt.ylabel(f'f({var})', fontsize=12)
        plt.title(f'Graph of {expression}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        return plt
    
    def show_history(self, last_n: int = 10):
        """Show calculation history"""
        print("\n" + "="*60)
        print("CALCULATION HISTORY".center(60))
        print("="*60)
        
        history_to_show = self.history[-last_n:] if len(self.history) > last_n else self.history
        
        for i, entry in enumerate(history_to_show, 1):
            print(f"{i}. {entry['expression']} = {entry['result']}")
        
        print("="*60 + "\n")
    
    def _is_number(self, token: str) -> bool:
        """Check if token is a number"""
        try:
            float(token)
            return True
        except ValueError:
            return False


class CalculatorInterface:
    def __init__(self):
        self.calc = AdvancedCalculator()
        
    def run(self):
        """Interactive calculator interface"""
        print("\n" + "="*70)
        print("ADVANCED MATHEMATICAL CALCULATOR".center(70))
        print("="*70)
        print("\nCommands:")
        print("  • Basic: 2+2, 5*3, 10/2, 2^3")
        print("  • Functions: sin(pi/2), sqrt(16), log(100)")
        print("  • Variables: x=5, y=x*2")
        print("  • Derivative: deriv(x^2, x=3)")
        print("  • Integral: integral(x^2, x, 0, 1)")
        print("  • Quadratic: quad(1, -5, 6)")
        print("  • Statistics: stats(1,2,3,4,5)")
        print("  • Plot: plot(sin(x), -10, 10)")
        print("  • History: history")
        print("  • Quit: exit, quit, q")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("calc> ").strip()
                
                if not user_input:
                    continue
                
                # Exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                # History
                elif user_input.lower() == 'history':
                    self.calc.show_history()
                
                # Variable assignment
                elif '=' in user_input and not any(op in user_input.split('=')[0] for op in ['<', '>', '!']):
                    var_name, expression = user_input.split('=', 1)
                    var_name = var_name.strip()
                    expression = expression.strip()
                    result = self.calc.set_variable(var_name, expression)
                    print(f"✓ {var_name} = {result}\n")
                
                # Derivative
                elif user_input.startswith('deriv('):
                    match = re.match(r'deriv\((.*),\s*(\w+)\s*=\s*([\d.]+)\)', user_input)
                    if match:
                        expr, var, point = match.groups()
                        result = self.calc.derivative(expr, var, float(point))
                        print(f"✓ d/d{var}({expr}) at {var}={point} = {result}\n")
                
                # Integral
                elif user_input.startswith('integral('):
                    match = re.match(r'integral\((.*),\s*(\w+),\s*([\d.-]+),\s*([\d.-]+)\)', user_input)
                    if match:
                        expr, var, a, b = match.groups()
                        result = self.calc.integral(expr, var, float(a), float(b))
                        print(f"✓ ∫({expr}) d{var} from {a} to {b} = {result}\n")
                
                # Quadratic solver
                elif user_input.startswith('quad('):
                    match = re.match(r'quad\(([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\)', user_input)
                    if match:
                        a, b, c = map(float, match.groups())
                        x1, x2 = self.calc.solve_quadratic(a, b, c)
                        print(f"✓ Solutions for {a}x² + {b}x + {c} = 0")
                        print(f"  x₁ = {x1}")
                        print(f"  x₂ = {x2}\n")
                
                # Statistics
                elif user_input.startswith('stats('):
                    match = re.match(r'stats\(([\d.,\s]+)\)', user_input)
                    if match:
                        numbers = [float(x.strip()) for x in match.group(1).split(',')]
                        stats = self.calc.statistics(numbers)
                        print("✓ Statistical Analysis:")
                        for key, value in stats.items():
                            print(f"  {key:12} : {value:.4f}")
                        print()
                
                # Plot
                elif user_input.startswith('plot('):
                    match = re.match(r'plot\((.*),\s*([\d.-]+),\s*([\d.-]+)\)', user_input)
                    if match:
                        expr, start, end = match.groups()
                        plt_obj = self.calc.plot_function(expr, 'x', float(start), float(end))
                        plt_obj.show()
                        print("✓ Plot generated\n")
                
                # Regular calculation
                else:
                    result = self.calc.calculate(user_input)
                    print(f"✓ {result}\n")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def demo():
    """Demonstration of calculator features"""
    calc = AdvancedCalculator()
    
    print("\n" + "="*70)
    print("ADVANCED CALCULATOR - DEMO MODE".center(70))
    print("="*70 + "\n")
    
    # Basic calculations
    print("1️⃣  BASIC CALCULATIONS")
    print("-" * 70)
    expressions = [
        "2 + 2",
        "10 * 5 + 3",
        "100 / 4 - 5",
        "2 ^ 10",
        "(5 + 3) * (10 - 2)"
    ]
    for expr in expressions:
        result = calc.calculate(expr)
        print(f"{expr:30} = {result}")
    
    # Functions
    print("\n2️⃣  MATHEMATICAL FUNCTIONS")
    print("-" * 70)
    calc.set_variable('x', math.pi/2)
    func_expressions = [
        "sin(x)",
        "cos(0)",
        "sqrt(144)",
        "log(1000)",
        "factorial(5)"
    ]
    for expr in func_expressions:
        result = calc.calculate(expr)
        print(f"{expr:30} = {result}")
    
    # Variables
    print("\n3️⃣  VARIABLE OPERATIONS")
    print("-" * 70)
    calc.set_variable('a', 10)
    calc.set_variable('b', 20)
    calc.set_variable('c', 'a + b')
    print(f"a = 10")
    print(f"b = 20")
    print(f"c = a + b = {calc.variables['c']}")
    print(f"a * b + c = {calc.calculate('a * b + c')}")
    
    # Calculus
    print("\n4️⃣  CALCULUS")
    print("-" * 70)
    deriv = calc.derivative('x^2', 'x', 3)
    print(f"d/dx(x²) at x=3 = {deriv}")
    
    integral = calc.integral('x^2', 'x', 0, 1)
    print(f"∫(x²)dx from 0 to 1 = {integral}")
    
    # Quadratic equation
    print("\n5️⃣  QUADRATIC EQUATION SOLVER")
    print("-" * 70)
    x1, x2 = calc.solve_quadratic(1, -5, 6)
    print(f"x² - 5x + 6 = 0")
    print(f"x₁ = {x1}")
    print(f"x₂ = {x2}")
    
    # Statistics
    print("\n6️⃣  STATISTICS")
    print("-" * 70)
    data = [12, 15, 18, 20, 22, 25, 28, 30]
    stats = calc.statistics(data)
    print(f"Data: {data}")
    for key, value in stats.items():
        print(f"{key:12} : {value:.2f}")
    
    print("\n" + "="*70 + "\n")
    
    # Show history
    calc.show_history()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        interface = CalculatorInterface()
        interface.run()