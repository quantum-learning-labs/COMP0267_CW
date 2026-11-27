"""Running circuits: Qiskit circuits from recipes, emulated devices and real hardware."""
import sys

try:  # IQM's compiler imports IPython.core.display_functions, which older IPython (e.g. Colab's) keeps in display
    import IPython.core.display_functions
except ImportError:
    try:
        import IPython.core.display
        sys.modules["IPython.core.display_functions"] = IPython.core.display
    except ImportError:
        pass
