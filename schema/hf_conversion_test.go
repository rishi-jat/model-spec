/*
 *     Copyright 2025 The CNCF ModelPack Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package schema_test

import (
	"fmt"
	"strings"
	"testing"
)

// convertHFToArchConfig converts HuggingFace config fields to architecture_config format.
// This mirrors the logic in tools/hf_to_arch.py for testing purposes.
func convertHFToArchConfig(hfConfig map[string]interface{}) (map[string]interface{}, error) {
	mappings := map[string]string{
		"numLayers":         "num_hidden_layers",
		"hiddenSize":        "hidden_size",
		"numAttentionHeads": "num_attention_heads",
	}

	archConfig := map[string]interface{}{
		"type": "transformer",
	}

	for archKey, hfKey := range mappings {
		val, ok := hfConfig[hfKey]
		if !ok {
			return nil, fmt.Errorf("missing required field: %s", hfKey)
		}
		// JSON numbers are float64 in Go
		numVal, ok := val.(float64)
		if !ok {
			return nil, fmt.Errorf("field %s must be a number", hfKey)
		}
		if numVal < 1 {
			return nil, fmt.Errorf("field %s must be >= 1", hfKey)
		}
		archConfig[archKey] = int(numVal)
	}

	return archConfig, nil
}

func TestHFToArchConfigConversionValid(t *testing.T) {
	hfConfig := map[string]interface{}{
		"num_hidden_layers":   float64(32),
		"hidden_size":         float64(4096),
		"num_attention_heads": float64(32),
		"vocab_size":          float64(32000), // extra field, should be ignored
	}

	archConfig, err := convertHFToArchConfig(hfConfig)
	if err != nil {
		t.Fatalf("conversion failed: %v", err)
	}

	if archConfig["type"] != "transformer" {
		t.Errorf("expected type=transformer, got %v", archConfig["type"])
	}
	if archConfig["numLayers"] != 32 {
		t.Errorf("expected numLayers=32, got %v", archConfig["numLayers"])
	}
	if archConfig["hiddenSize"] != 4096 {
		t.Errorf("expected hiddenSize=4096, got %v", archConfig["hiddenSize"])
	}
	if archConfig["numAttentionHeads"] != 32 {
		t.Errorf("expected numAttentionHeads=32, got %v", archConfig["numAttentionHeads"])
	}
}

func TestHFToArchConfigConversionMissingField(t *testing.T) {
	hfConfig := map[string]interface{}{
		"hidden_size":         float64(4096),
		"num_attention_heads": float64(32),
		// num_hidden_layers missing
	}

	_, err := convertHFToArchConfig(hfConfig)
	if err == nil {
		t.Fatalf("expected conversion to fail for missing field")
	}
	if !strings.Contains(err.Error(), "num_hidden_layers") {
		t.Errorf("error should mention missing field, got: %v", err)
	}
}

func TestHFToArchConfigConversionInvalidType(t *testing.T) {
	hfConfig := map[string]interface{}{
		"num_hidden_layers":   "32", // string instead of number
		"hidden_size":         float64(4096),
		"num_attention_heads": float64(32),
	}

	_, err := convertHFToArchConfig(hfConfig)
	if err == nil {
		t.Fatalf("expected conversion to fail for invalid type")
	}
	if !strings.Contains(err.Error(), "must be a number") {
		t.Errorf("error should mention type issue, got: %v", err)
	}
}

func TestHFToArchConfigConversionZeroValue(t *testing.T) {
	hfConfig := map[string]interface{}{
		"num_hidden_layers":   float64(0), // invalid: must be >= 1
		"hidden_size":         float64(4096),
		"num_attention_heads": float64(32),
	}

	_, err := convertHFToArchConfig(hfConfig)
	if err == nil {
		t.Fatalf("expected conversion to fail for zero value")
	}
	if !strings.Contains(err.Error(), ">= 1") {
		t.Errorf("error should mention minimum value, got: %v", err)
	}
}
