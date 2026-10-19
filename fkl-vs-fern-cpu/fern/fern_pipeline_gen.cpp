// Fern (gern) generator for the Figure 1 pipeline of one region:
//   NV12 frame --(nv12_to_rgb on the region's rectangle = crop)--> crop (h x w float3)
//              --(resize_bilinear)--> resized (32 x 64 float3) --(letterbox)--> padded (64 x 64 float3)
//              --(normalize)--> out (64 x 64 float3, one plane of the N x 64 x 64 x 3 tensor)
// The data structures and subroutines are annotated here (what a Fern user writes for Fern); their C++ is in
// fern_impl.h. The region (x, y, w, h) is a data structure (Rect) whose fields appear in the subset expressions as
// ADTMember (rect["x"] -> rect.x in the generated code): run-time data, so one generated function serves every
// region size and position, with no recompilation.
//
// Usage: fern_pipeline_gen <output directory>
// Writes one C++ source per program (generated/ has the checked-in copy, regenerate.cmake rebuilds and compares it),
// each defining void <name>(NV12View& frame, Rect& rect, ImageF3& out) (a template on the tile size) and Fern's
// calling convention, extern "C" void hook_<name>(void** args). Nothing is compiled or dlopened: the benchmark's build
// compiles the sources with the flags of FKL's code.

#include "annotations/abstract_function.h"
#include "codegen/codegen.h"
#include "compose/composable.h"

#include <fstream>
#include <optional>
#include <iostream>
#include <string>

using namespace gern;

namespace annot {

// NV12 frame, subsets are rectangles of pixels {row, col, rows, cols}
class NV12Frame : public AbstractDataType {
public:
    explicit NV12Frame(const std::string &name) : name(name) {}
    std::string getName() const override { return name; }
    std::string getType() const override { return "fern_bvf::NV12View"; }
    std::vector<Variable> getFields() const override { return {r, c, l_r, l_c}; }
    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{.name = "fern_bvf::NV12View::allocate", .args = {r, c, l_r, l_c}};
    }
    FunctionSignature getFreeFunction() const override { return FunctionSignature{.name = "destroy", .args = {}}; }
    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{.name = "insert", .args = {r, c, l_r, l_c}};
    }
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{.name = "query", .args = {r, c, l_r, l_c}};
    }

private:
    std::string name;
    Variable r{"r"}, c{"c"}, l_r{"l_r"}, l_c{"l_c"};
};

// The region of one call; only its fields are used (in subset expressions)
class RegionRect : public AbstractDataType {
public:
    explicit RegionRect(const std::string &name) : name(name) {}
    std::string getName() const override { return name; }
    std::string getType() const override { return "fern_bvf::Rect"; }
    std::vector<Variable> getFields() const override { return {i}; }
    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{.name = "fern_bvf::Rect::allocate", .args = {i}};
    }
    FunctionSignature getFreeFunction() const override { return FunctionSignature{.name = "destroy", .args = {}}; }
    FunctionSignature getInsertFunction() const override { return FunctionSignature{.name = "insert", .args = {i}}; }
    FunctionSignature getQueryFunction() const override { return FunctionSignature{.name = "query", .args = {i}}; }

private:
    std::string name;
    Variable i{"i"};
};

// float3 image, subsets {row, col, rows, cols}; queries are views (no insert), intermediates are allocated
class ImageF3 : public AbstractDataType {
public:
    explicit ImageF3(const std::string &name) : name(name) {}
    std::string getName() const override { return name; }
    std::string getType() const override { return "fern_bvf::ImageF3"; }
    std::vector<Variable> getFields() const override { return {x, y, l_x, l_y}; }
    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{.name = "fern_bvf::ImageF3::allocate", .args = {x, y, l_x, l_y}};
    }
    FunctionSignature getFreeFunction() const override { return FunctionSignature{.name = "destroy", .args = {}}; }
    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{.name = "insert", .args = {x, y, l_x, l_y}};
    }
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{.name = "query", .args = {x, y, l_x, l_y}};
    }
    bool insertQuery() const override { return false; }

private:
    std::string name;
    Variable x{"x"}, y{"y"}, l_x{"l_x"}, l_y{"l_y"};
};

// Common shape of the four subroutines: tileable over the rows and columns of their output
class ImageFunction : public AbstractFunction {
public:
    ImageFunction(std::string fn, AbstractDataTypePtr in, AbstractDataTypePtr out, AbstractDataTypePtr rect)
        : fn(std::move(fn)), input(in), output(out), rect(rect) {}
    Annotation getAnnotation() override {
        Variable x("x"), y("y"), l_x("l_x"), l_y("l_y");
        return annotate(Tileable(x = Expr(0), output["row"], l_x,
                                 Tileable(y = Expr(0), output["col"], l_y,
                                          Produces::Subset(output, {x, y, l_x, l_y}),
                                          consumes(x, y, l_x, l_y))));
    }
    FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = fn;
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }
    std::vector<std::string> getHeader() override { return {"fern_impl.h"}; }

protected:
    virtual Consumes consumes(Variable x, Variable y, Variable l_x, Variable l_y) = 0;
    std::string fn;
    AbstractDataTypePtr input, output, rect;
};

// Output pixel (x, y) of the crop needs frame pixel (x + rect.y, y + rect.x): the crop is the subset's offset
class ConvertNV12ToRGB : public ImageFunction {
public:
    ConvertNV12ToRGB(AbstractDataTypePtr frame, AbstractDataTypePtr crop, AbstractDataTypePtr rect)
        : ImageFunction("fern_bvf::nv12_to_rgb", frame, crop, rect) {}

protected:
    Consumes consumes(Variable x, Variable y, Variable l_x, Variable l_y) override {
        return Consumes::Subset(input, {x + rect["y"], y + rect["x"], l_x, l_y});
    }
};

// Any output tile needs the whole crop (rect.h x rect.w): a whole-image subroutine
class ResizeBilinear : public ImageFunction {
public:
    ResizeBilinear(AbstractDataTypePtr crop, AbstractDataTypePtr resized, AbstractDataTypePtr rect)
        : ImageFunction("fern_bvf::resize_bilinear", crop, resized, rect) {}

protected:
    Consumes consumes(Variable, Variable, Variable, Variable) override {
        return Consumes::Subset(input, {Expr(0), Expr(0), rect["h"], rect["w"]});
    }
};

// Any band of output rows needs the whole 32 x 64 resized image (the band's rows inside it are not an affine
// function of the band: Fern's subset expressions have no min/max)
class Letterbox : public ImageFunction {
public:
    Letterbox(AbstractDataTypePtr resized, AbstractDataTypePtr padded)
        : ImageFunction("fern_bvf::letterbox", resized, padded, AbstractDataTypePtr()) {}

protected:
    Consumes consumes(Variable, Variable, Variable, Variable) override {
        return Consumes::Subset(input, {Expr(0), Expr(0), Expr(32), Expr(64)});
    }
};

class Normalize : public ImageFunction {
public:
    Normalize(AbstractDataTypePtr padded, AbstractDataTypePtr out)
        : ImageFunction("fern_bvf::normalize", padded, out, AbstractDataTypePtr()) {}

protected:
    Consumes consumes(Variable x, Variable y, Variable l_x, Variable l_y) override {
        return Consumes::Subset(input, {x, y, l_x, l_y});
    }
};

} // namespace annot

// The data structures and subroutines of one program (fresh objects per program)
struct Pipeline {
    AbstractDataTypePtr frame{new const annot::NV12Frame("frame")};
    AbstractDataTypePtr rect{new const annot::RegionRect("rect")};
    AbstractDataTypePtr crop{new const annot::ImageF3("crop")};
    AbstractDataTypePtr resized{new const annot::ImageF3("resized")};
    AbstractDataTypePtr padded{new const annot::ImageF3("padded")};
    AbstractDataTypePtr out{new const annot::ImageF3("out")};
    annot::ConvertNV12ToRGB convert{frame, crop, rect};
    annot::ResizeBilinear resize{crop, resized, rect};
    annot::Letterbox letterbox{resized, padded};
    annot::Normalize normalize{padded, out};
    std::vector<Parameter> parameters() const { return {Parameter(frame), Parameter(rect), Parameter(out)}; }
};

static void emit(const std::string &dir, const std::string &name, const Composable &program,
                 const std::optional<std::vector<Parameter>> &parameters) {
    codegen::CodeGenerator cg(parameters, name);
    codegen::CGStmt code = cg.generate_code(program);
    const std::string path = dir + "/" + name + ".cpp";
    std::ofstream f(path);
    f << "// Generated by fern_pipeline_gen (gern) - do not edit\n" << code;
    std::cout << "wrote " << path << " (arguments:";
    for (const auto &a : cg.getArgumentOrder()) std::cout << " " << a;
    std::cout << ")\n";
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <output directory>\n";
        return 1;
    }
    const std::string dir = argv[1];

    // The schedule of the benchmark: the whole chain fused under one loop over bands of 16 output rows. Fern queries
    // the frame at (rect.y, rect.x) (the crop), allocates the crop (rect.h x rect.w) and resized (32 x 64)
    // intermediates and runs nv12_to_rgb and resize_bilinear once (its finalizer hoists them out of the band loop:
    // they do not depend on the band), then per band allocates a band of the padded intermediate, letterboxes into it
    // and normalizes it into the output. The band height is a compile-time constant (a template parameter of the
    // generated function) that must divide the 64 output rows: Fern never clamps the last tile.
    {
        Pipeline p;
        Variable t("t");
        emit(dir, "fern_band16",
             Composable({Tile(p.out["row"], t.bind(16))(p.convert(p.frame, p.crop), p.resize(p.crop, p.resized),
                                                         p.letterbox(p.resized, p.padded),
                                                         p.normalize(p.padded, p.out))}),
             p.parameters());
    }

    // Pitfalls, kept as evidence (not used by the benchmark):
    {   // No Tile at the top: Fern treats the program's inputs as already restricted to the subsets it consumes and
        // passes the whole frame to nv12_to_rgb without the (rect.y, rect.x) query: the crop offset is lost and the
        // code converts the top-left corner of the frame (wrong results, no error)
        Pipeline p;
        emit(dir, "pitfall_untiled",
             Composable({p.convert(p.frame, p.crop), p.resize(p.crop, p.resized), p.letterbox(p.resized, p.padded),
                         p.normalize(p.padded, p.out)}),
             p.parameters());
    }
    {   // Same when only the tail is under the Tile
        Pipeline p;
        Variable t("t");
        emit(dir, "pitfall_tail_tiled",
             Composable({p.convert(p.frame, p.crop), p.resize(p.crop, p.resized),
                         Tile(p.out["row"], t.bind(16))(p.letterbox(p.resized, p.padded), p.normalize(p.padded, p.out))}),
             p.parameters());
    }
    {   // The region as run-time scalars (ox, oy, w, h function arguments) instead of fields of a data structure:
        // the scalars inside subset expressions lower to 0 (src/codegen/concretize.cpp:346-347) and the generated
        // function does not even take them as arguments
        class ConvertScalars : public annot::ImageFunction {
        public:
            ConvertScalars(AbstractDataTypePtr frame, AbstractDataTypePtr crop)
                : ImageFunction("fern_bvf::nv12_to_rgb_scalars", frame, crop, AbstractDataTypePtr()) {}
            FunctionSignature getFunction() override {
                FunctionSignature f;
                f.name = fn;
                f.args = {Parameter(input), Parameter(output), Parameter(ox), Parameter(oy)};
                return f;
            }

        protected:
            Consumes consumes(Variable x, Variable y, Variable l_x, Variable l_y) override {
                return Consumes::Subset(input, {x + oy, y + ox, l_x, l_y});
            }
            Variable ox{"ox"}, oy{"oy"};
        };
        class ResizeScalars : public annot::ImageFunction {
        public:
            ResizeScalars(AbstractDataTypePtr crop, AbstractDataTypePtr resized)
                : ImageFunction("fern_bvf::resize_bilinear_scalars", crop, resized, AbstractDataTypePtr()) {}
            FunctionSignature getFunction() override {
                FunctionSignature f;
                f.name = fn;
                f.args = {Parameter(input), Parameter(output), Parameter(w), Parameter(h)};
                return f;
            }

        protected:
            Consumes consumes(Variable, Variable, Variable, Variable) override {
                return Consumes::Subset(input, {Expr(0), Expr(0), h, w});
            }
            Variable w{"w"}, h{"h"};
        };
        Pipeline p;
        ConvertScalars convert(p.frame, p.crop);
        ResizeScalars resize(p.crop, p.resized);
        Variable t("t"), ox("ox"), oy("oy"), w("w"), h("h");
        emit(dir, "pitfall_scalar_region",
             Composable({Tile(p.out["row"], t.bind(16))(convert(p.frame, p.crop, ox, oy),
                                                         resize(p.crop, p.resized, w, h),
                                                         p.letterbox(p.resized, p.padded),
                                                         p.normalize(p.padded, p.out))}),
             std::nullopt);
    }
    return 0;
}
