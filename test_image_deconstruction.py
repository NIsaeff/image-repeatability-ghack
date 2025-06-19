#!/usr/bin/env python3
"""
Test script for enhanced image deconstruction with FLUX.1-schnell optimization.
"""

import os
import sys
from image_deconstruction import ImageDeconstructor

def test_image_analysis():
    """Test the enhanced image analysis capabilities."""
    
    # Check if environment variables are set
    friendli_token = os.environ.get("FRIENDLI_TOKEN")
    endpoint_id = os.environ.get("FRIENDLI_ENDPOINT_ID")
    
    if not friendli_token or not endpoint_id:
        print("Error: Please set FRIENDLI_TOKEN and FRIENDLI_ENDPOINT_ID environment variables")
        print("Example:")
        print("export FRIENDLI_TOKEN='your_token_here'")
        print("export FRIENDLI_ENDPOINT_ID='your_endpoint_id_here'")
        return
    
    # Initialize the deconstructor
    deconstructor = ImageDeconstructor(friendli_token, endpoint_id)
    
    # Test scenarios
    test_scenarios = [
        "on a wooden table in a cozy kitchen",
        "floating in space with stars in the background",
        "on a sandy beach with ocean waves",
        "in a modern office with glass walls",
        "in a forest with sunlight filtering through trees"
    ]
    
    # Get image path from user
    print("Enhanced Image Deconstruction Test for FLUX.1-schnell")
    print("=" * 60)
    
    image_path = input("Enter the path to your test image: ").strip()
    
    if not image_path or not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return
    
    try:
        # Read image as bytes
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        print(f"\nAnalyzing image: {image_path}")
        print(f"Image size: {len(image_bytes)} bytes")
        
        # Step 1: Analyze the image
        print("\n" + "="*50)
        print("STEP 1: IMAGE ANALYSIS")
        print("="*50)
        
        analysis = deconstructor.analyze_image_for_objects(image_bytes)
        
        # Display analysis results
        print("\nAnalysis Results:")
        print(f"  Object Description: {analysis.get('object_description', 'N/A')}")
        print(f"  Size: {analysis.get('size', 'N/A')}")
        print(f"  Aspect Ratio: {analysis.get('aspect_ratio', 'N/A'):.2f}")
        print(f"  Brightness: {analysis.get('brightness', 'N/A')}")
        print(f"  Contrast: {analysis.get('contrast', 'N/A')}")
        print(f"  Composition: {', '.join(analysis.get('composition', []))}")
        print(f"  Text Detected: {analysis.get('text', 'N/A')}")
        print(f"  Objects Detected: {', '.join(analysis.get('objects', []))}")
        
        # Shape analysis
        shape_analysis = analysis.get('shape_analysis', {})
        print(f"  Dominant Shape: {shape_analysis.get('dominant_shape', 'N/A')}")
        print(f"  Shape Count: {shape_analysis.get('shape_count', 'N/A')}")
        
        # Material analysis
        material_analysis = analysis.get('material_analysis', {})
        print(f"  Material Type: {material_analysis.get('material_type', 'N/A')}")
        print(f"  Texture: {material_analysis.get('texture', 'N/A')}")
        print(f"  Reflectivity: {material_analysis.get('reflectivity', 'N/A')}")
        
        # Lighting analysis
        lighting_analysis = analysis.get('lighting_analysis', {})
        print(f"  Lighting Type: {lighting_analysis.get('lighting_type', 'N/A')}")
        print(f"  Shadow Intensity: {lighting_analysis.get('shadow_intensity', 'N/A')}")
        
        # Step 2: Generate images for different scenarios
        print("\n" + "="*50)
        print("STEP 2: GENERATING IMAGES")
        print("="*50)
        
        # Ask user if they want to generate images
        generate_images = input("\nDo you want to generate images for different scenarios? (y/n): ").strip().lower()
        
        if generate_images == 'y':
            # Use first 3 scenarios for testing
            test_scenarios_subset = test_scenarios[:3]
            
            print(f"\nGenerating images for {len(test_scenarios_subset)} scenarios...")
            
            results = deconstructor.generate_consistent_object_series(image_bytes, test_scenarios_subset)
            
            print("\nGeneration Results:")
            print("="*30)
            
            for scenario, result in results['generated_images'].items():
                print(f"\nScenario: {scenario}")
                print(f"  Prompt: {result['prompt']}")
                if result['url']:
                    print(f"  Image URL: {result['url']}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*50)
        print("TEST COMPLETED")
        print("="*50)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_prompt_generation():
    """Test the FLUX-optimized prompt generation."""
    
    # Mock analysis data for testing
    mock_analysis = {
        "object_description": "red cylindrical metallic object with text 'COLA'",
        "brightness": "medium",
        "contrast": "high",
        "material_analysis": {
            "material_type": "metallic",
            "texture": "smooth",
            "reflectivity": "glossy"
        },
        "lighting_analysis": {
            "lighting_type": "soft_lighting",
            "shadow_intensity": "soft_shadows"
        },
        "text": "COLA",
        "objects": ["red cylindrical metallic object"],
        "shape_analysis": {
            "dominant_shape": "circle",
            "shape_count": 1
        }
    }
    
    # Initialize deconstructor (without real credentials for this test)
    deconstructor = ImageDeconstructor("test_token", "test_endpoint")
    
    test_scenarios = [
        "on a wooden table",
        "floating in space",
        "in a modern kitchen"
    ]
    
    print("FLUX-optimized Prompt Generation Test")
    print("=" * 50)
    
    for scenario in test_scenarios:
        prompt = deconstructor.generate_flux_optimized_prompt(mock_analysis, scenario)
        print(f"\nScenario: {scenario}")
        print(f"Generated Prompt: {prompt}")
        print(f"Prompt Length: {len(prompt)} characters")

if __name__ == "__main__":
    print("Enhanced Image Deconstruction Test Suite")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "prompt":
        test_prompt_generation()
    else:
        test_image_analysis() 
