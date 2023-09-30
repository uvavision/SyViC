using UnityEditor;
using System.IO;

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Object = UnityEngine.Object;
// using System.IO;
// using System.Collections.Generic;
using System.Text.RegularExpressions;
// using System.Collections;
using UnityEditor.Animations;

// using System.Collections.Generic;
// using System.Text;
// using System.Linq;
// using UnityEngine.UI;
using UnityEditor.AnimatedValues;
using UnityEngine.SceneManagement;

using UnityEditor.Presets;
using System.Reflection;

public class ChangeModelToHumanoid : AssetPostprocessor
{
    void OnPreprocessModel()
    {
        ModelImporter modelImporter = assetImporter as ModelImporter;
        modelImporter.animationType =   ModelImporterAnimationType.Human;
        EditorUtility.SetDirty(modelImporter);
        AssetDatabase.SaveAssets();        
    }
}