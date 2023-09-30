using UnityEditor;
using System.IO;

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Object = UnityEngine.Object;
using System.Text.RegularExpressions;
using UnityEditor.Animations;

using UnityEditor.AnimatedValues;
using UnityEngine.SceneManagement;

public class CreateAssetBundles
{
    [MenuItem("Assets/Build Linux AssetBundles")]
    static void BuildAllAssetBundlesLinux()
    {
        string assetBundleDirectory = "Assets/AssetBundles/StandaloneLinux64";
        if(!Directory.Exists(assetBundleDirectory))
        {
            Directory.CreateDirectory(assetBundleDirectory);
        }
        BuildPipeline.BuildAssetBundles(assetBundleDirectory, 
                                        BuildAssetBundleOptions.None, 
                                        BuildTarget.StandaloneLinux64);
    }

    [MenuItem("Assets/Build MacOS AssetBundles")]
    static void BuildAllAssetBundlesMac()
    {
        string assetBundleDirectory = "Assets/AssetBundles/StandaloneOSX";
        if(!Directory.Exists(assetBundleDirectory))
        {
            Directory.CreateDirectory(assetBundleDirectory);
        }
        BuildPipeline.BuildAssetBundles(assetBundleDirectory, 
                                        BuildAssetBundleOptions.None, 
                                        BuildTarget.StandaloneOSX);
    }

    [MenuItem("Assets/Generate Materials")]
    static void GenerateMaterials()
    {
        string[] all_paths_tmp = new [] {"Assets/Resources/Textures/SURREAL/female/", "Assets/Resources/Textures/SURREAL/male/", "Assets/Resources/Textures/people_snapshot_public/", "Assets/Resources/Textures/Multi-Garmentdataset/" };

        foreach (string all_textures_path in all_paths_tmp) 
        {
            DirectoryInfo d = new DirectoryInfo(all_textures_path); //Assuming Test is your Folder
            FileInfo[] Files = d.GetFiles("*.jpg"); //Getting Text files
            string str = "";
            foreach(FileInfo file in Files )
            {
                Material newMaterial = new Material(Shader.Find("Unlit/Texture"));

                byte[] fileData = File.ReadAllBytes(all_textures_path + file.Name);
                Texture2D tex = new Texture2D(2, 2);
                tex.LoadImage(fileData); //..this will auto-resize the texture dimensions.
                newMaterial.mainTexture = tex;

                string newAssetName = all_textures_path + file.Name.Replace(".jpg", ".mat");
                Debug.Log (newAssetName);
                
                AssetDatabase.CreateAsset(newMaterial, newAssetName);
                AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(file.Name.Replace(".jpg", ""), "");
                AssetDatabase.SaveAssets();

                fileData = File.ReadAllBytes(all_textures_path + file.Name);
                tex = new Texture2D(2, 2);
                tex.LoadImage(fileData);
                newMaterial.mainTexture = tex;
            }
        }
    }

    [MenuItem("Assets/Delete Meta Materials Files")]
    static void DeleteMaterials()
    {

        DirectoryInfo d = new DirectoryInfo(@"Assets/Resources/Textures/SURREAL/female/"); //Assuming Test is your Folder
        FileInfo[] Files = d.GetFiles("*.mat"); //Getting Text files
        string str = "";
        foreach(FileInfo file in Files )
        {
            File.Delete(@"Assets/Resources/Textures/SURREAL/female/" + file.Name);
            // File.Delete(@"Assets/Resources/Textures/SURREAL/female/" + file.Name + ".meta");
        }
    }
    
    // black	Solid black. RGBA is (0, 0, 0, 1).
    // blue	Solid blue. RGBA is (0, 0, 1, 1).
    // clear	Completely transparent. RGBA is (0, 0, 0, 0).
    // cyan	Cyan. RGBA is (0, 1, 1, 1).
    // gray	Gray. RGBA is (0.5, 0.5, 0.5, 1).
    // green	Solid green. RGBA is (0, 1, 0, 1).
    // grey	English spelling for gray. RGBA is the same (0.5, 0.5, 0.5, 1).
    // magenta	Magenta. RGBA is (1, 0, 1, 1).
    // red	Solid red. RGBA is (1, 0, 0, 1).
    // white	Solid white. RGBA is (1, 1, 1, 1).
    // yellow  Yellow. RGBA is (1, 0.92, 0.016, 1), but the color is nice to look at!
    [MenuItem("Assets/Save Color Materials")]
    static void AssetsToBeSaved()
    {
        Color[] colorList = new[] { Color.red, Color.green, Color.blue, Color.black, Color.gray, Color.cyan, Color.magenta, Color.white, Color.yellow  };
        string[] colorNames = new[] { "red", "green", "blue", "black", "gray", "cyan", "magenta", "white", "yellow" };
        for (int i = 0; i < colorList.Length; ++i)
        {
            Material material = new Material(Shader.Find("Specular"));
            var materialName = "material_" + colorNames[i]  + ".mat";
            var newAssetName = "Assets/Materials/" + materialName;
            AssetDatabase.CreateAsset(material, newAssetName);
            AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(materialName.Replace(".mat", ""), "");

            material.SetColor("_Color", colorList[i]);
        }

        AssetDatabase.SaveAssets();
    }

    [MenuItem("Assets/Save More Color Materials")]
    static void WideVarietyOfColorsToBeSaved()
    {
        string[] colorNames = new[] { "aliceblue", "antiquewhite", "antiquewhite1", "antiquewhite2", "antiquewhite3", "antiquewhite4", "aqua", "aquamarine1", "aquamarine2", "aquamarine3", "aquamarine4", "azure1", "azure2", "azure3", "azure4", "banana", "beige", "bisque1", "bisque2", "bisque3", "bisque4", "black", "blanchedalmond", "blue", "blue2", "blue3", "blue4", "blueviolet", "brick", "brown", "brown1", "brown2", "brown3", "brown4", "burlywood", "burlywood1", "burlywood2", "burlywood3", "burlywood4", "burntsienna", "burntumber", "cadetblue", "cadetblue1", "cadetblue2", "cadetblue3", "cadetblue4", "cadmiumorange", "cadmiumyellow", "carrot", "chartreuse1", "chartreuse2", "chartreuse3", "chartreuse4", "chocolate", "chocolate1", "chocolate2", "chocolate3", "chocolate4", "cobalt", "cobaltgreen", "coldgrey", "coral", "coral1", "coral2", "coral3", "coral4", "cornflowerblue", "cornsilk1", "cornsilk2", "cornsilk3", "cornsilk4", "crimson", "cyan2", "cyan3", "cyan4", "darkgoldenrod", "darkgoldenrod1", "darkgoldenrod2", "darkgoldenrod3", "darkgoldenrod4", "darkgray", "darkgreen", "darkkhaki", "darkolivegreen", "darkolivegreen1", "darkolivegreen2", "darkolivegreen3", "darkolivegreen4", "darkorange", "darkorange1", "darkorange2", "darkorange3", "darkorange4", "darkorchid", "darkorchid1", "darkorchid2", "darkorchid3", "darkorchid4", "darksalmon", "darkseagreen", "darkseagreen1", "darkseagreen2", "darkseagreen3", "darkseagreen4", "darkslateblue", "darkslategray", "darkslategray1", "darkslategray2", "darkslategray3", "darkslategray4", "darkturquoise", "darkviolet", "deeppink1", "deeppink2", "deeppink3", "deeppink4", "deepskyblue1", "deepskyblue2", "deepskyblue3", "deepskyblue4", "dimgray", "dodgerblue1", "dodgerblue2", "dodgerblue3", "dodgerblue4", "eggshell", "emeraldgreen", "firebrick", "firebrick1", "firebrick2", "firebrick3", "firebrick4", "flesh", "floralwhite", "forestgreen", "gainsboro", "ghostwhite", "gold1", "gold2", "gold3", "gold4", "goldenrod", "goldenrod1", "goldenrod2", "goldenrod3", "goldenrod4", "gray", "gray1", "gray10", "gray11", "gray12", "gray13", "gray14", "gray15", "gray16", "gray17", "gray18", "gray19", "gray2", "gray20", "gray21", "gray22", "gray23", "gray24", "gray25", "gray26", "gray27", "gray28", "gray29", "gray3", "gray30", "gray31", "gray32", "gray33", "gray34", "gray35", "gray36", "gray37", "gray38", "gray39", "gray4", "gray40", "gray42", "gray43", "gray44", "gray45", "gray46", "gray47", "gray48", "gray49", "gray5", "gray50", "gray51", "gray52", "gray53", "gray54", "gray55", "gray56", "gray57", "gray58", "gray59", "gray6", "gray60", "gray61", "gray62", "gray63", "gray64", "gray65", "gray66", "gray67", "gray68", "gray69", "gray7", "gray70", "gray71", "gray72", "gray73", "gray74", "gray75", "gray76", "gray77", "gray78", "gray79", "gray8", "gray80", "gray81", "gray82", "gray83", "gray84", "gray85", "gray86", "gray87", "gray88", "gray89", "gray9", "gray90", "gray91", "gray92", "gray93", "gray94", "gray95", "gray97", "gray98", "gray99", "green", "green1", "green2", "green3", "green4", "greenyellow", "honeydew1", "honeydew2", "honeydew3", "honeydew4", "hotpink", "hotpink1", "hotpink2", "hotpink3", "hotpink4", "indianred", "indianred1", "indianred2", "indianred3", "indianred4", "indigo", "ivory1", "ivory2", "ivory3", "ivory4", "ivoryblack", "khaki", "khaki1", "khaki2", "khaki3", "khaki4", "lavender", "lavenderblush1", "lavenderblush2", "lavenderblush3", "lavenderblush4", "lawngreen", "lemonchiffon1", "lemonchiffon2", "lemonchiffon3", "lemonchiffon4", "lightblue", "lightblue1", "lightblue2", "lightblue3", "lightblue4", "lightcoral", "lightcyan1", "lightcyan2", "lightcyan3", "lightcyan4", "lightgoldenrod1", "lightgoldenrod2", "lightgoldenrod3", "lightgoldenrod4", "lightgoldenrodyellow", "lightgrey", "lightpink", "lightpink1", "lightpink2", "lightpink3", "lightpink4", "lightsalmon1", "lightsalmon2", "lightsalmon3", "lightsalmon4", "lightseagreen", "lightskyblue", "lightskyblue1", "lightskyblue2", "lightskyblue3", "lightskyblue4", "lightslateblue", "lightslategray", "lightsteelblue", "lightsteelblue1", "lightsteelblue2", "lightsteelblue3", "lightsteelblue4", "lightyellow1", "lightyellow2", "lightyellow3", "lightyellow4", "limegreen", "linen", "magenta", "magenta2", "magenta3", "magenta4", "manganeseblue", "maroon", "maroon1", "maroon2", "maroon3", "maroon4", "mediumorchid", "mediumorchid1", "mediumorchid2", "mediumorchid3", "mediumorchid4", "mediumpurple", "mediumpurple1", "mediumpurple2", "mediumpurple3", "mediumpurple4", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "melon", "midnightblue", "mint", "mintcream", "mistyrose1", "mistyrose2", "mistyrose3", "mistyrose4", "moccasin", "navajowhite1", "navajowhite2", "navajowhite3", "navajowhite4", "navy", "oldlace", "olive", "olivedrab", "olivedrab1", "olivedrab2", "olivedrab3", "olivedrab4", "orange", "orange1", "orange2", "orange3", "orange4", "orangered1", "orangered2", "orangered3", "orangered4", "orchid", "orchid1", "orchid2", "orchid3", "orchid4", "palegoldenrod", "palegreen", "palegreen1", "palegreen2", "palegreen3", "palegreen4", "paleturquoise1", "paleturquoise2", "paleturquoise3", "paleturquoise4", "palevioletred", "palevioletred1", "palevioletred2", "palevioletred3", "palevioletred4", "papayawhip", "peachpuff1", "peachpuff2", "peachpuff3", "peachpuff4", "peacock", "pink", "pink1", "pink2", "pink3", "pink4", "plum", "plum1", "plum2", "plum3", "plum4", "powderblue", "purple", "purple1", "purple2", "purple3", "purple4", "raspberry", "rawsienna", "red1", "red2", "red3", "red4", "rosybrown", "rosybrown1", "rosybrown2", "rosybrown3", "rosybrown4", "royalblue", "royalblue1", "royalblue2", "royalblue3", "royalblue4", "salmon", "salmon1", "salmon2", "salmon3", "salmon4", "sandybrown", "sapgreen", "seagreen1", "seagreen2", "seagreen3", "seagreen4", "seashell1", "seashell2", "seashell3", "seashell4", "sepia", "sgibeet", "sgibrightgray", "sgichartreuse", "sgidarkgray", "sgigray12", "sgigray16", "sgigray32", "sgigray36", "sgigray52", "sgigray56", "sgigray72", "sgigray76", "sgigray92", "sgigray96", "sgilightblue", "sgilightgray", "sgiolivedrab", "sgisalmon", "sgislateblue", "sgiteal", "sienna", "sienna1", "sienna2", "sienna3", "sienna4", "silver", "skyblue", "skyblue1", "skyblue2", "skyblue3", "skyblue4", "slateblue", "slateblue1", "slateblue2", "slateblue3", "slateblue4", "slategray", "slategray1", "slategray2", "slategray3", "slategray4", "snow1", "snow2", "snow3", "snow4", "springgreen", "springgreen1", "springgreen2", "springgreen3", "steelblue", "steelblue1", "steelblue2", "steelblue3", "steelblue4", "tan", "tan1", "tan2", "tan3", "tan4", "teal", "thistle", "thistle1", "thistle2", "thistle3", "thistle4", "tomato1", "tomato2", "tomato3", "tomato4", "turquoise", "turquoise1", "turquoise2", "turquoise3", "turquoise4", "turquoiseblue", "violet", "violetred", "violetred1", "violetred2", "violetred3", "violetred4", "warmgrey", "wheat", "wheat1", "wheat2", "wheat3", "wheat4", "white", "whitesmoke", "yellow1", "yellow2", "yellow3", "yellow4" };
        string[] colorValues = new[] {"#f0f8ff", "#faebd7", "#ffefdb", "#eedfcc", "#cdc0b0", "#8b8378", "#00ffff", "#7fffd4", "#76eec6", "#66cdaa", "#458b74", "#f0ffff", "#e0eeee", "#c1cdcd", "#838b8b", "#e3cf57", "#f5f5dc", "#ffe4c4", "#eed5b7", "#cdb79e", "#8b7d6b", "#000000", "#ffebcd", "#0000ff", "#0000ee", "#0000cd", "#00008b", "#8a2be2", "#9c661f", "#a52a2a", "#ff4040", "#ee3b3b", "#cd3333", "#8b2323", "#deb887", "#ffd39b", "#eec591", "#cdaa7d", "#8b7355", "#8a360f", "#8a3324", "#5f9ea0", "#98f5ff", "#8ee5ee", "#7ac5cd", "#53868b", "#ff6103", "#ff9912", "#ed9121", "#7fff00", "#76ee00", "#66cd00", "#458b00", "#d2691e", "#ff7f24", "#ee7621", "#cd661d", "#8b4513", "#3d59ab", "#3d9140", "#808a87", "#ff7f50", "#ff7256", "#ee6a50", "#cd5b45", "#8b3e2f", "#6495ed", "#fff8dc", "#eee8cd", "#cdc8b1", "#8b8878", "#dc143c", "#00eeee", "#00cdcd", "#008b8b", "#b8860b", "#ffb90f", "#eead0e", "#cd950c", "#8b6508", "#a9a9a9", "#006400", "#bdb76b", "#556b2f", "#caff70", "#bcee68", "#a2cd5a", "#6e8b3d", "#ff8c00", "#ff7f00", "#ee7600", "#cd6600", "#8b4500", "#9932cc", "#bf3eff", "#b23aee", "#9a32cd", "#68228b", "#e9967a", "#8fbc8f", "#c1ffc1", "#b4eeb4", "#9bcd9b", "#698b69", "#483d8b", "#2f4f4f", "#97ffff", "#8deeee", "#79cdcd", "#528b8b", "#00ced1", "#9400d3", "#ff1493", "#ee1289", "#cd1076", "#8b0a50", "#00bfff", "#00b2ee", "#009acd", "#00688b", "#696969", "#1e90ff", "#1c86ee", "#1874cd", "#104e8b", "#fce6c9", "#00c957", "#b22222", "#ff3030", "#ee2c2c", "#cd2626", "#8b1a1a", "#ff7d40", "#fffaf0", "#228b22", "#dcdcdc", "#f8f8ff", "#ffd700", "#eec900", "#cdad00", "#8b7500", "#daa520", "#ffc125", "#eeb422", "#cd9b1d", "#8b6914", "#808080", "#030303", "#1a1a1a", "#1c1c1c", "#1f1f1f", "#212121", "#242424", "#262626", "#292929", "#2b2b2b", "#2e2e2e", "#303030", "#050505", "#333333", "#363636", "#383838", "#3b3b3b", "#3d3d3d", "#404040", "#424242", "#454545", "#474747", "#4a4a4a", "#080808", "#4d4d4d", "#4f4f4f", "#525252", "#545454", "#575757", "#595959", "#5c5c5c", "#5e5e5e", "#616161", "#636363", "#0a0a0a", "#666666", "#6b6b6b", "#6e6e6e", "#707070", "#737373", "#757575", "#787878", "#7a7a7a", "#7d7d7d", "#0d0d0d", "#7f7f7f", "#828282", "#858585", "#878787", "#8a8a8a", "#8c8c8c", "#8f8f8f", "#919191", "#949494", "#969696", "#0f0f0f", "#999999", "#9c9c9c", "#9e9e9e", "#a1a1a1", "#a3a3a3", "#a6a6a6", "#a8a8a8", "#ababab", "#adadad", "#b0b0b0", "#121212", "#b3b3b3", "#b5b5b5", "#b8b8b8", "#bababa", "#bdbdbd", "#bfbfbf", "#c2c2c2", "#c4c4c4", "#c7c7c7", "#c9c9c9", "#141414", "#cccccc", "#cfcfcf", "#d1d1d1", "#d4d4d4", "#d6d6d6", "#d9d9d9", "#dbdbdb", "#dedede", "#e0e0e0", "#e3e3e3", "#171717", "#e5e5e5", "#e8e8e8", "#ebebeb", "#ededed", "#f0f0f0", "#f2f2f2", "#f7f7f7", "#fafafa", "#fcfcfc", "#008000", "#00ff00", "#00ee00", "#00cd00", "#008b00", "#adff2f", "#f0fff0", "#e0eee0", "#c1cdc1", "#838b83", "#ff69b4", "#ff6eb4", "#ee6aa7", "#cd6090", "#8b3a62", "#cd5c5c", "#ff6a6a", "#ee6363", "#cd5555", "#8b3a3a", "#4b0082", "#fffff0", "#eeeee0", "#cdcdc1", "#8b8b83", "#292421", "#f0e68c", "#fff68f", "#eee685", "#cdc673", "#8b864e", "#e6e6fa", "#fff0f5", "#eee0e5", "#cdc1c5", "#8b8386", "#7cfc00", "#fffacd", "#eee9bf", "#cdc9a5", "#8b8970", "#add8e6", "#bfefff", "#b2dfee", "#9ac0cd", "#68838b", "#f08080", "#e0ffff", "#d1eeee", "#b4cdcd", "#7a8b8b", "#ffec8b", "#eedc82", "#cdbe70", "#8b814c", "#fafad2", "#d3d3d3", "#ffb6c1", "#ffaeb9", "#eea2ad", "#cd8c95", "#8b5f65", "#ffa07a", "#ee9572", "#cd8162", "#8b5742", "#20b2aa", "#87cefa", "#b0e2ff", "#a4d3ee", "#8db6cd", "#607b8b", "#8470ff", "#778899", "#b0c4de", "#cae1ff", "#bcd2ee", "#a2b5cd", "#6e7b8b", "#ffffe0", "#eeeed1", "#cdcdb4", "#8b8b7a", "#32cd32", "#faf0e6", "#ff00ff", "#ee00ee", "#cd00cd", "#8b008b", "#03a89e", "#800000", "#ff34b3", "#ee30a7", "#cd2990", "#8b1c62", "#ba55d3", "#e066ff", "#d15fee", "#b452cd", "#7a378b", "#9370db", "#ab82ff", "#9f79ee", "#8968cd", "#5d478b", "#3cb371", "#7b68ee", "#00fa9a", "#48d1cc", "#c71585", "#e3a869", "#191970", "#bdfcc9", "#f5fffa", "#ffe4e1", "#eed5d2", "#cdb7b5", "#8b7d7b", "#ffe4b5", "#ffdead", "#eecfa1", "#cdb38b", "#8b795e", "#000080", "#fdf5e6", "#808000", "#6b8e23", "#c0ff3e", "#b3ee3a", "#9acd32", "#698b22", "#ff8000", "#ffa500", "#ee9a00", "#cd8500", "#8b5a00", "#ff4500", "#ee4000", "#cd3700", "#8b2500", "#da70d6", "#ff83fa", "#ee7ae9", "#cd69c9", "#8b4789", "#eee8aa", "#98fb98", "#9aff9a", "#90ee90", "#7ccd7c", "#548b54", "#bbffff", "#aeeeee", "#96cdcd", "#668b8b", "#db7093", "#ff82ab", "#ee799f", "#cd6889", "#8b475d", "#ffefd5", "#ffdab9", "#eecbad", "#cdaf95", "#8b7765", "#33a1c9", "#ffc0cb", "#ffb5c5", "#eea9b8", "#cd919e", "#8b636c", "#dda0dd", "#ffbbff", "#eeaeee", "#cd96cd", "#8b668b", "#b0e0e6", "#800080", "#9b30ff", "#912cee", "#7d26cd", "#551a8b", "#872657", "#c76114", "#ff0000", "#ee0000", "#cd0000", "#8b0000", "#bc8f8f", "#ffc1c1", "#eeb4b4", "#cd9b9b", "#8b6969", "#4169e1", "#4876ff", "#436eee", "#3a5fcd", "#27408b", "#fa8072", "#ff8c69", "#ee8262", "#cd7054", "#8b4c39", "#f4a460", "#308014", "#54ff9f", "#4eee94", "#43cd80", "#2e8b57", "#fff5ee", "#eee5de", "#cdc5bf", "#8b8682", "#5e2612", "#8e388e", "#c5c1aa", "#71c671", "#555555", "#1e1e1e", "#282828", "#515151", "#5b5b5b", "#848484", "#8e8e8e", "#b7b7b7", "#c1c1c1", "#eaeaea", "#f4f4f4", "#7d9ec0", "#aaaaaa", "#8e8e38", "#c67171", "#7171c6", "#388e8e", "#a0522d", "#ff8247", "#ee7942", "#cd6839", "#8b4726", "#c0c0c0", "#87ceeb", "#87ceff", "#7ec0ee", "#6ca6cd", "#4a708b", "#6a5acd", "#836fff", "#7a67ee", "#6959cd", "#473c8b", "#708090", "#c6e2ff", "#b9d3ee", "#9fb6cd", "#6c7b8b", "#fffafa", "#eee9e9", "#cdc9c9", "#8b8989", "#00ff7f", "#00ee76", "#00cd66", "#008b45", "#4682b4", "#63b8ff", "#5cacee", "#4f94cd", "#36648b", "#d2b48c", "#ffa54f", "#ee9a49", "#cd853f", "#8b5a2b", "#008080", "#d8bfd8", "#ffe1ff", "#eed2ee", "#cdb5cd", "#8b7b8b", "#ff6347", "#ee5c42", "#cd4f39", "#8b3626", "#40e0d0", "#00f5ff", "#00e5ee", "#00c5cd", "#00868b", "#00c78c", "#ee82ee", "#d02090", "#ff3e96", "#ee3a8c", "#cd3278", "#8b2252", "#808069", "#f5deb3", "#ffe7ba", "#eed8ae", "#cdba96", "#8b7e66", "#ffffff", "#f5f5f5", "#ffff00", "#eeee00", "#cdcd00", "#8b8b00"};
        for (int i = 0; i < colorNames.Length; ++i)
        {
            Material material = new Material(Shader.Find("Specular"));
            var materialName = "material_" + colorNames[i]  + ".mat";
            var newAssetName = "Assets/Materials/" + materialName;
            AssetDatabase.CreateAsset(material, newAssetName);
            AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(materialName.Replace(".mat", ""), "");

            Color newColor;
            ColorUtility.TryParseHtmlString(colorValues[i], out newColor);
            material.SetColor("_Color", newColor);
        }

        AssetDatabase.SaveAssets();
    }


    [MenuItem("Assets/Extract Animations and create controllers")]
    static void ExtractAnimationsFromFBX()
    {
        DirectoryInfo d = new DirectoryInfo("Assets/Resources"); //Assuming Test is your Folder

        FileInfo[] Files = d.GetFiles("*.fbx"); //Getting Text files

        foreach(FileInfo file in Files )
        {
            string str =file.Name;
            // Debug.Log (str);

            //Extract the animations from the fbxs and place the in resources to be
            //used for overrideAnimation        
            AnimationClip orgClip = (AnimationClip)AssetDatabase.LoadAssetAtPath("Assets/Resources/" + file.Name,typeof(AnimationClip) );
            SerializedObject serializedClip = new SerializedObject(orgClip);
            //Save the clip
            AnimationClip placeClip = new AnimationClip();
            if( !Resources.Load("Assets/Resources/" + file.Name.Replace(".fbx", ".anim")))
            {
                var newAssetName = "Assets/Resources/" + file.Name.Replace(".fbx", ".anim");
                EditorUtility.CopySerialized(orgClip,placeClip);
                AssetDatabase.CreateAsset(placeClip, newAssetName);
                AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(file.Name.Replace(".fbx", "_anim"), "");
                AssetDatabase.Refresh();

                Debug.Log (newAssetName);

                var newControllerName = "Assets/Resources/" + file.Name.Replace(".fbx", ".controller");
                AnimatorController anim_controller = UnityEditor.Animations.AnimatorController.CreateAnimatorControllerAtPathWithClip(newControllerName, placeClip);
                // AssetDatabase.CreateAsset(anim_controller, newControllerName);
                AssetImporter.GetAtPath(newControllerName).SetAssetBundleNameAndVariant(file.Name.Replace(".fbx", "_controller"), "");
                AssetDatabase.Refresh();

                Debug.Log (newControllerName);
            }
        }
    }

    [MenuItem("Assets/Create SMPL Animations")]
    static void CreateSMPLAnimations()
    {
        DirectoryInfo d = new DirectoryInfo("Assets/Resources"); //Assuming Test is your Folder
        
        FileInfo[] Files = d.GetFiles("*_male.controller"); //Getting Text files
        GameObject cmu_male_smplx = (GameObject)AssetDatabase.LoadAssetAtPath("Assets/SMPLX_base/cmu_male.prefab", typeof(GameObject) );

        Animator anim_ref = cmu_male_smplx.GetComponent<Animator>();

        if (anim_ref == null) {
                Debug.Log("Failed to load AssetBundle!");
                return;
        }

        foreach(FileInfo file in Files )
        {
            string str =file.Name;
            Debug.Log(str);

            // set animation
            AnimatorController animController = (AnimatorController)AssetDatabase.LoadAssetAtPath("Assets/Resources/" + file.Name, typeof(AnimatorController) );
            // Animator anim_ref = cmu_male_smplx.GetComponent<Animator>();
            anim_ref.runtimeAnimatorController = animController;

            // set material 
            DirectoryInfo dMat = new DirectoryInfo("Assets/Resources/Textures/Multi-Garmentdataset"); //Assuming Test is your Folder
            FileInfo[] FilesMat = dMat.GetFiles("*.mat"); //Getting Text files
            foreach(FileInfo fileMat in FilesMat )
            {
                var materialName = fileMat.Name.Replace(".mat", "");
                Transform[] transforms = cmu_male_smplx.GetComponentsInChildren<Transform>();
                foreach(Transform t in transforms)
                {
                    if (t.gameObject.name == "SMPLX-mesh-male")
                    {
                        SkinnedMeshRenderer renderer = t.gameObject.GetComponentInChildren<SkinnedMeshRenderer>();

                        Material tmp_mat = Resources.Load("Textures/Multi-Garmentdataset/" + materialName, typeof(Material)) as Material;
                        renderer.sharedMaterial = tmp_mat;

                        Material[] mats = renderer.sharedMaterials;
                        mats[0] = tmp_mat;
                        renderer.sharedMaterials = mats;
                    }
                }

                // save new asset
                var newAssetName = "Assets/Animated_SMPLX/" + file.Name.Replace(".controller", "") + "_" + materialName + ".prefab";
                if(!AssetDatabase.CopyAsset("Assets/SMPLX_base/cmu_male.prefab", newAssetName))
                    Debug.LogWarning("Failed to copy");
                AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(file.Name.Replace(".controller", "") + "_" + materialName + "_smplx", "");
            }
        }
    }
    
    [MenuItem("Assets/CLOTHE SMPLs")]
    static void ClotheSMPLs()
    {

        GameObject cmu_male_smplx = (GameObject)AssetDatabase.LoadAssetAtPath("Assets/SMPLX_base/cmu_male.prefab", typeof(GameObject) );
        Debug.Log("GOT PREFAB");

        DirectoryInfo dMat = new DirectoryInfo("Assets/Resources/Textures/Multi-Garmentdataset"); //Assuming Test is your Folder
        FileInfo[] FilesMat = dMat.GetFiles("*.mat"); //Getting Text files
        foreach(FileInfo fileMat in FilesMat )
        {
            // var materialName = "125611494277906_registered_tex";
            var materialName = fileMat.Name.Replace(".mat", "");
            Debug.Log("GOT MAT NAME");
            Debug.Log(materialName);

            Transform[] transforms = cmu_male_smplx.GetComponentsInChildren<Transform>();
            foreach(Transform t in transforms)
            {
                if (t.gameObject.name == "SMPLX-mesh-male")
                {
                    SkinnedMeshRenderer renderer = t.gameObject.GetComponentInChildren<SkinnedMeshRenderer>();
                    Material tmp_mat = Resources.Load("Textures/Multi-Garmentdataset/" + materialName, typeof(Material)) as Material;
                    renderer.sharedMaterial = tmp_mat;

                    Material[] mats = renderer.sharedMaterials;
                    mats[0] = tmp_mat;
                    renderer.sharedMaterials = mats;
                }
            }

            // save new asset
            var newAssetName = "Assets/Animated_SMPLX/" + materialName + ".prefab";
            if(!AssetDatabase.CopyAsset("Assets/SMPLX_base/cmu_male.prefab", newAssetName))
                Debug.LogWarning("Failed to copy");
            AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(materialName + "_smplx", "");
        }
    }

    [MenuItem("Assets/CLOTHE SMPLs using TDW Prefab")]
    static void ClotheTDWPrefabSMPLs()
    {
        GameObject cmu_male_smplx = (GameObject)AssetDatabase.LoadAssetAtPath("Assets/SMPLX_base/PREFABS/cmu_37_37_01_male/cmu_37_37_01_male.prefab", typeof(GameObject) );
        Debug.Log("GOT PREFAB");

        DirectoryInfo dMat = new DirectoryInfo("Assets/Resources/Textures/Multi-Garmentdataset"); //Assuming Test is your Folder
        FileInfo[] FilesMat = dMat.GetFiles("*.mat"); //Getting Text files
        foreach(FileInfo fileMat in FilesMat )
        {
            var materialName = fileMat.Name.Replace(".mat", "");
            Debug.Log("GOT MAT NAME");
            Debug.Log(materialName);

            Transform[] transforms = cmu_male_smplx.GetComponentsInChildren<Transform>();
            foreach(Transform t in transforms)
            {
                if (t.gameObject.name == "SMPLX-mesh-male")
                {
                    // SkinnedMeshRenderer renderer = t.gameObject.GetComponentInChildren<SkinnedMeshRenderer>();
                    Renderer renderer = t.gameObject.GetComponent<Renderer>();
                    Material tmp_mat = Resources.Load("Textures/Multi-Garmentdataset/" + materialName, typeof(Material)) as Material;
                    renderer.material = tmp_mat;
                }
            }

            // save new asset
            System.IO.Directory.CreateDirectory("Assets/SMPLX_base/PREFABS/" + materialName);
            var newAssetName = "Assets/SMPLX_base/PREFABS/" + materialName + "/" + materialName + ".prefab";
            if(!AssetDatabase.CopyAsset("Assets/SMPLX_base/PREFABS/cmu_37_37_01_male/cmu_37_37_01_male.prefab", newAssetName))
                Debug.LogWarning("Failed to copy");
            // probably not necessary since we are using the TDW assetbundler
            AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(materialName + "_smplx", "");
        }
    }

}
