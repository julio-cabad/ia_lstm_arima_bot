import unittest
from test_indicators import TestIndicators

def run_all_tests():
    # Ejecutar pruebas de indicadores
    print("\n=== Ejecutando pruebas de indicadores ===")
    tester = TestIndicators()
    tester.test_all_indicators()

if __name__ == "__main__":
    run_all_tests() 